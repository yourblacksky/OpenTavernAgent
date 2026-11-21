//! Batch only across independent games. Seats within a game retain their original action order.
use crate::{
    backend::Execution,
    net::PolicyValueNet,
    observation::{Observation, observe_seat_with_rules},
    policy,
    rollout::{self, GameRollout, Trajectory, Transition},
    rules::{ActionCounts, ActionHistory, PolicyRules},
};
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use taverngym::{Game, RulesConfig};

struct Running {
    game: Game,
    seed: u64,
    cursor: usize,
    rng: Pcg64Mcg,
    histories: Vec<ActionHistory>,
    seat_opponents: Vec<Option<usize>>,
    actions: ActionCounts,
    trajectories: Vec<Trajectory>,
}
impl Running {
    fn new(
        seed: u64,
        cfg: &RulesConfig,
        rules: PolicyRules,
        pool_len: usize,
        pool_prob: f64,
    ) -> Result<Self, String> {
        let mut game = Game::new(seed, cfg.clone());
        crate::heroes::select_heroes_where(&mut game, seed, |_| !rules.learn_hero_select)?;
        let count = game.players.len();
        let seat_opponents = rollout::assign_opponent_seats(seed, count, pool_len, pool_prob);
        Ok(Self {
            game,
            seed,
            cursor: 0,
            rng: rollout::game_rng(seed),
            histories: vec![ActionHistory::default(); count],
            trajectories: (0..count)
                .map(|seat| Trajectory {
                    seed,
                    seat,
                    placement: 0,
                    terminal_health: 0.0,
                    is_learner: seat_opponents[seat].is_none(),
                    steps: Vec::new(),
                })
                .collect(),
            seat_opponents,
            actions: ActionCounts::default(),
        })
    }
    /// Advance any historical-opponent turns on CPU and stop at the next learner decision (or game end).
    fn observe_until_learner(
        &mut self,
        rules: PolicyRules,
        opponents: &[PolicyValueNet],
    ) -> Result<Option<(usize, Observation)>, String> {
        loop {
            if self.game.stopped_reason.is_some() {
                return Err(format!("batched simulator stopped: seed={}", self.seed));
            }
            if self.game.is_over() {
                return Ok(None);
            }
            let mut next = None;
            for _ in 0..self.game.players.len() {
                let seat = self.cursor;
                self.cursor = (seat + 1) % self.game.players.len();
                if self.game.needs_action(seat) {
                    next = Some(seat);
                    break;
                }
            }
            let Some(seat) = next else {
                return Err(format!("batched simulator stalled: seed={}", self.seed));
            };
            let obs = observe_seat_with_rules(&self.game, seat, rules, &self.histories[seat]);
            if let Some(opp_idx) = self.seat_opponents[seat] {
                let turn = self.game.turn;
                let decision = policy::decide(&opponents[opp_idx], &obs, &mut self.rng, false)?;
                crate::actions::apply_with_rules(&mut self.game, seat, decision.action, rules)
                    .map_err(str::to_owned)?;
                self.actions
                    .record(&self.histories[seat], turn, decision.action);
                self.histories[seat].record(turn, decision.action);
                continue;
            }
            return Ok(Some((seat, obs)));
        }
    }
    fn apply(
        &mut self,
        seat: usize,
        obs: Observation,
        logits: &[f32],
        value: f32,
        rules: PolicyRules,
    ) -> Result<(), String> {
        let decision = policy::from_logits(logits, value, &obs.mask, &mut self.rng, false)?;
        let turn = self.game.turn;
        let health = (self.game.players[seat].health + self.game.players[seat].armor).max(0) as f32;
        let tier = self.game.players[seat].tier;
        crate::actions::apply_with_rules(&mut self.game, seat, decision.action, rules)
            .map_err(str::to_owned)?;
        self.actions
            .record(&self.histories[seat], turn, decision.action);
        self.histories[seat].record(turn, decision.action);
        if self.trajectories[seat].is_learner {
            self.trajectories[seat].steps.push(Transition {
                turn,
                health,
                tier,
                obs,
                action: decision.action,
                old_log_prob: decision.log_prob,
                old_value: value,
                reward: 0.,
                terminated: false,
            });
        }
        Ok(())
    }
    fn finish(mut self) -> Result<GameRollout, String> {
        for t in &mut self.trajectories {
            t.placement = self.game.players[t.seat].placement;
            let p = &self.game.players[t.seat];
            t.terminal_health = if p.alive {
                (p.health + p.armor).max(0) as f32
            } else {
                0.0
            };
            if t.placement == 0 {
                return Err("batched rollout missing placement".into());
            }
            if let Some(last) = t.steps.last_mut() {
                last.terminated = true;
                last.reward =
                    taverngym::env::placement_reward(t.placement, self.game.players.len());
            }
        }
        Ok(GameRollout {
            trajectories: self.trajectories,
            turns: self.game.turn,
            turn_limit_end: self.game.turn_limit_end,
            truncated_combats: self.game.truncated_combats,
            actions: self.actions,
        })
    }
}

pub fn collect(
    net: &PolicyValueNet,
    seeds: &[u64],
    cfg: &RulesConfig,
    rules: PolicyRules,
    pool: &rayon::ThreadPool,
    execution: &Execution,
) -> Result<Vec<GameRollout>, String> {
    collect_with_pool(net, &[], 0.0, seeds, cfg, rules, pool, execution)
}

#[allow(clippy::too_many_arguments)]
pub fn collect_with_pool(
    net: &PolicyValueNet,
    opponents: &[PolicyValueNet],
    pool_prob: f64,
    seeds: &[u64],
    cfg: &RulesConfig,
    rules: PolicyRules,
    pool: &rayon::ThreadPool,
    execution: &Execution,
) -> Result<Vec<GameRollout>, String> {
    cfg.validate()?;
    let mut games = pool.install(|| {
        seeds
            .par_iter()
            .map(|&seed| Running::new(seed, cfg, rules, opponents.len(), pool_prob))
            .collect::<Result<Vec<_>, _>>()
    })?;
    loop {
        let ready = pool.install(|| {
            games
                .par_iter_mut()
                .map(|game| game.observe_until_learner(rules, opponents))
                .collect::<Result<Vec<_>, _>>()
        })?;
        let observations: Vec<_> = ready
            .iter()
            .filter_map(|r| r.as_ref().map(|(_, o)| o))
            .collect();
        if observations.is_empty() {
            break;
        }
        let (logits, values) = execution.infer(net, &observations)?;
        let mut row = 0;
        let work: Vec<_> = ready
            .into_iter()
            .map(|pending| {
                pending.map(|(seat, obs)| {
                    let result = (seat, obs, logits.row(row).to_vec(), values[row]);
                    row += 1;
                    result
                })
            })
            .collect();
        pool.install(|| {
            games
                .par_iter_mut()
                .zip(work.into_par_iter())
                .try_for_each(|(game, work)| {
                    if let Some((seat, obs, logits, value)) = work {
                        game.apply(seat, obs, &logits, value, rules)
                    } else {
                        Ok(())
                    }
                })
        })?;
    }
    games.into_iter().map(Running::finish).collect()
}
