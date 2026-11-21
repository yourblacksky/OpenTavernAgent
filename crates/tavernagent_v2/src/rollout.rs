//! Complete, unsplit games from a fixed policy; each seat owns its own trajectory.

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use taverngym::{Game, RulesConfig};

use crate::gae::{self, ValueStep};
use crate::heroes::select_heroes_where;
use crate::net::PolicyValueNet;
use crate::observation::{Observation, observe_seat_with_rules};
use crate::policy;
use crate::rules::{ActionCounts, ActionHistory, PolicyRules};

/// Potential-based reward shaping coefficients: F(s, s') = gamma * Phi(s') - Phi(s),
/// with Phi(s_terminal) = 0 so the optimal policy is preserved (Ng et al. 1999).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RewardShaping {
    pub health_coef: f32,
    pub tier_coef: f32,
}

impl RewardShaping {
    pub fn is_active(self) -> bool {
        self.health_coef != 0.0 || self.tier_coef != 0.0
    }

    pub fn potential(self, health: f32, tier: u8) -> f32 {
        let h = (health / 40.0).clamp(0.0, 1.0);
        let t = (tier.saturating_sub(1) as f32 / 5.0).clamp(0.0, 1.0);
        self.health_coef * h + self.tier_coef * t
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Transition {
    pub turn: u32,
    pub health: f32,
    pub tier: u8,
    pub obs: Observation,
    pub action: usize,
    pub old_log_prob: f64,
    pub old_value: f32,
    pub reward: f32,
    pub terminated: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Trajectory {
    pub seed: u64,
    pub seat: usize,
    pub placement: u8,
    pub terminal_health: f32,
    /// True when this seat was controlled by the current learning policy (on-policy).
    pub is_learner: bool,
    pub steps: Vec<Transition>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    pub transition: Transition,
    pub advantage: f32,
    pub return_: f32,
    pub terminal_reward: f32,
    pub future_health: Option<f32>,
}

impl Trajectory {
    pub fn into_samples(self, gamma: f32, lambda: f32) -> Result<Vec<Sample>, String> {
        self.into_samples_with_trace(gamma, lambda, None)
    }

    pub fn into_samples_with_trace(
        self,
        gamma: f32,
        lambda: f32,
        within_turn_lambda: Option<f32>,
    ) -> Result<Vec<Sample>, String> {
        self.into_samples_with_shaping(gamma, lambda, within_turn_lambda, RewardShaping::default())
    }

    pub fn into_samples_with_shaping(
        self,
        gamma: f32,
        lambda: f32,
        within_turn_lambda: Option<f32>,
        shaping: RewardShaping,
    ) -> Result<Vec<Sample>, String> {
        if let Some(last) = self.steps.last()
            && (!last.terminated
                || self.steps[..self.steps.len() - 1]
                    .iter()
                    .any(|s| s.terminated))
        {
            return Err("complete trajectory must end in exactly one terminal transition".into());
        }
        let shaping_active = shaping.is_active();
        let potentials: Vec<f32> = if shaping_active {
            self.steps
                .iter()
                .map(|s| shaping.potential(s.health, s.tier))
                .collect()
        } else {
            Vec::new()
        };
        let values: Vec<_> = self
            .steps
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let shaped_reward = if shaping_active {
                    let phi_next = if s.terminated || i + 1 >= potentials.len() {
                        0.0
                    } else {
                        potentials[i + 1]
                    };
                    s.reward + gamma * phi_next - potentials[i]
                } else {
                    s.reward
                };
                ValueStep {
                    value: s.old_value,
                    reward: shaped_reward,
                    terminated: s.terminated,
                }
            })
            .collect();
        let terminal_reward = self.steps.last().map_or(0.0, |s| s.reward);
        let mut health_by_turn = std::collections::BTreeMap::new();
        for step in &self.steps {
            health_by_turn.entry(step.turn).or_insert(step.health);
        }
        let future_health: Vec<_> = self
            .steps
            .iter()
            .map(|step| {
                health_by_turn
                    .range(step.turn.saturating_add(3)..)
                    .next()
                    .map(|(_, h)| *h)
                    .or(if self.terminal_health <= 0.0 {
                        Some(0.0)
                    } else {
                        None
                    })
                    .map(|h| 2.0 * (h / 60.0).clamp(0.0, 1.0) - 1.0)
            })
            .collect();
        let targets = if let Some(within) = within_turn_lambda {
            let mut trace = vec![lambda; self.steps.len()];
            for (i, pair) in self.steps.windows(2).enumerate() {
                let elapsed = pair[1]
                    .turn
                    .checked_sub(pair[0].turn)
                    .ok_or("trajectory turns must be monotone")?;
                trace[i] = if elapsed == 0 {
                    within
                } else {
                    lambda.powf(elapsed as f32)
                };
            }
            gae::estimate_with_trace(&values, 0.0, gamma, &trace)?
        } else {
            gae::estimate(&values, 0.0, gamma, lambda)?
        };
        Ok(self
            .steps
            .into_iter()
            .zip(targets)
            .zip(future_health)
            .map(
                |((transition, (advantage, return_)), future_health)| Sample {
                    transition,
                    advantage,
                    return_,
                    terminal_reward,
                    future_health,
                },
            )
            .collect())
    }
}

#[derive(Debug, PartialEq)]
pub struct GameRollout {
    pub trajectories: Vec<Trajectory>,
    pub turns: u32,
    pub turn_limit_end: bool,
    pub truncated_combats: u32,
    pub actions: ActionCounts,
}

pub fn game_rng(seed: u64) -> Pcg64Mcg {
    Pcg64Mcg::seed_from_u64(seed ^ 0x5050_4F32)
}

pub fn collect_game(
    net: &PolicyValueNet,
    seed: u64,
    cfg: &RulesConfig,
) -> Result<GameRollout, String> {
    collect_game_with_rules(net, seed, cfg, PolicyRules::default())
}

/// Assign each seat either `None` (current learning policy) or `Some(idx)` into `opponents`.
/// Seat 0 is always the current learning policy so every game produces on-policy samples.
pub fn assign_opponent_seats(
    seed: u64,
    num_players: usize,
    pool_len: usize,
    pool_prob: f64,
) -> Vec<Option<usize>> {
    let mut assigned = vec![None; num_players];
    if pool_len == 0 || pool_prob <= 0.0 {
        return assigned;
    }
    let mut rng = Pcg64Mcg::seed_from_u64(seed ^ 0x4F50_5050);
    for slot in assigned.iter_mut().skip(1) {
        if rng.random::<f64>() < pool_prob {
            *slot = Some(rng.random_range(0..pool_len));
        }
    }
    assigned
}

pub fn collect_game_with_rules(
    net: &PolicyValueNet,
    seed: u64,
    cfg: &RulesConfig,
    rules: PolicyRules,
) -> Result<GameRollout, String> {
    collect_game_with_pool(net, &[], 0.0, seed, cfg, rules)
}

pub fn collect_game_with_pool(
    net: &PolicyValueNet,
    opponents: &[PolicyValueNet],
    pool_prob: f64,
    seed: u64,
    cfg: &RulesConfig,
    rules: PolicyRules,
) -> Result<GameRollout, String> {
    cfg.validate()?;
    let mut game = Game::new(seed, cfg.clone());
    select_heroes_where(&mut game, seed, |_| !rules.learn_hero_select)?;
    let seat_opponents =
        assign_opponent_seats(seed, game.players.len(), opponents.len(), pool_prob);
    let mut rng = game_rng(seed);
    let mut histories = vec![ActionHistory::default(); game.players.len()];
    let mut actions = ActionCounts::default();
    let mut trajectories: Vec<_> = (0..game.players.len())
        .map(|seat| Trajectory {
            seed,
            seat,
            placement: 0,
            terminal_health: 0.0,
            is_learner: seat_opponents[seat].is_none(),
            steps: Vec::new(),
        })
        .collect();
    while !game.is_over() {
        if let Some(reason) = &game.stopped_reason {
            return Err(format!("simulator stopped: seed={seed}, reason={reason:?}"));
        }
        let mut acted = false;
        // Observe immediately before each action. Other seats may alter public state.
        for (seat, trajectory) in trajectories.iter_mut().enumerate() {
            if !game.needs_action(seat) {
                continue;
            }
            let turn = game.turn;
            let health = (game.players[seat].health + game.players[seat].armor).max(0) as f32;
            let tier = game.players[seat].tier;
            let obs = observe_seat_with_rules(&game, seat, rules, &histories[seat]);
            let seat_net = match seat_opponents[seat] {
                Some(idx) => &opponents[idx],
                None => net,
            };
            let decision = policy::decide(seat_net, &obs, &mut rng, false)
                .map_err(|e| format!("seed={seed}, seat={seat}, turn={}: {e}", game.turn))?;
            crate::actions::apply_with_rules(&mut game, seat, decision.action, rules).map_err(
                |e| format!("seed={seed}, seat={seat}, action={}: {e}", decision.action),
            )?;
            actions.record(&histories[seat], turn, decision.action);
            histories[seat].record(turn, decision.action);
            if trajectory.is_learner {
                trajectory.steps.push(Transition {
                    turn,
                    health,
                    tier,
                    obs,
                    action: decision.action,
                    old_log_prob: decision.log_prob,
                    old_value: decision.value,
                    reward: 0.0,
                    terminated: false,
                });
            }
            acted = true;
        }
        if !acted {
            return Err(format!(
                "self-play stalled: seed={seed}, turn={}",
                game.turn
            ));
        }
    }
    if game.stopped_reason.is_some() {
        return Err(format!("simulator stopped: seed={seed}"));
    }
    for trajectory in &mut trajectories {
        let placement = game.players[trajectory.seat].placement;
        if placement == 0 {
            return Err(format!("missing terminal placement: seed={seed}"));
        }
        trajectory.placement = placement;
        let p = &game.players[trajectory.seat];
        trajectory.terminal_health = if p.alive {
            (p.health + p.armor).max(0) as f32
        } else {
            0.0
        };
        // Elimination may happen while this seat is waiting. Its final action receives
        // the reward, irrespective of which seat triggered the combat or elimination.
        if let Some(last) = trajectory.steps.last_mut() {
            last.reward = taverngym::env::placement_reward(placement, game.players.len());
            last.terminated = true;
        }
    }
    Ok(GameRollout {
        trajectories,
        turns: game.turn,
        turn_limit_end: game.turn_limit_end,
        truncated_combats: game.truncated_combats,
        actions,
    })
}

pub fn collect(
    net: &PolicyValueNet,
    seeds: &[u64],
    cfg: &RulesConfig,
    pool: &rayon::ThreadPool,
) -> Result<Vec<GameRollout>, String> {
    collect_with_rules(net, seeds, cfg, pool, PolicyRules::default())
}

pub fn collect_with_rules(
    net: &PolicyValueNet,
    seeds: &[u64],
    cfg: &RulesConfig,
    pool: &rayon::ThreadPool,
    rules: PolicyRules,
) -> Result<Vec<GameRollout>, String> {
    collect_with_pool(net, &[], 0.0, seeds, cfg, pool, rules)
}

pub fn collect_with_pool(
    net: &PolicyValueNet,
    opponents: &[PolicyValueNet],
    pool_prob: f64,
    seeds: &[u64],
    cfg: &RulesConfig,
    pool: &rayon::ThreadPool,
    rules: PolicyRules,
) -> Result<Vec<GameRollout>, String> {
    // Indexed collection and a separate RNG per game keep results independent of workers.
    pool.install(|| {
        seeds
            .par_iter()
            .map(|&s| collect_game_with_pool(net, opponents, pool_prob, s, cfg, rules))
            .collect()
    })
}

/// Normalize only the actor advantages, once for the entire fresh rollout batch.
pub fn normalize_advantages(samples: &mut [Sample]) -> Result<(), String> {
    if samples.is_empty() || samples.iter().any(|s| !s.advantage.is_finite()) {
        return Err("advantage normalization needs finite, nonempty samples".into());
    }
    let mean = samples.iter().map(|s| s.advantage as f64).sum::<f64>() / samples.len() as f64;
    let variance = samples
        .iter()
        .map(|s| (s.advantage as f64 - mean).powi(2))
        .sum::<f64>()
        / samples.len() as f64;
    let scale = variance.sqrt().max(1e-8);
    for sample in samples {
        sample.advantage = ((sample.advantage as f64 - mean) / scale) as f32;
    }
    Ok(())
}
