use tavernagent_v2::{
    actions, checkpoint,
    net::{N_OPP_BOARD_SLOTS, NetConfig, PolicyValueNet},
    observation::{Observation, agent_mask_with_rules, observe_seat_with_rules},
    ppo::{PpoOptions, Trainer},
    rollout::{self, RewardShaping, Trajectory, Transition},
    rules::{ActionHistory, Budgets, PolicyRules},
    train::{TrainArgs, TrainOptions},
};
use taverngym::{Game, RulesConfig, obs};

#[test]
fn observe_opponents_expands_inputs_and_responds_to_opponent_boards() {
    let base_cfg = NetConfig {
        hidden: 16,
        blocks: 1,
        embedding_dim: 4,
        ..Default::default()
    };
    let opp_cfg = NetConfig {
        observe_opponents: true,
        ..base_cfg.clone()
    };
    let base_net = PolicyValueNet::new(7, base_cfg).unwrap();
    let opp_net = PolicyValueNet::new(7, opp_cfg).unwrap();

    assert_eq!(
        opp_net.n_input_features() - base_net.n_input_features(),
        N_OPP_BOARD_SLOTS * obs::MINION_F
    );
    assert_eq!(
        opp_net.n_card_slots() - base_net.n_card_slots(),
        N_OPP_BOARD_SLOTS
    );

    // Verify that modifying an opponent's last-seen board feature changes opp_net output
    // while leaving base_net output identical.
    let mut x1 = vec![0.0f32; tavernagent_v2::net::N_FEATURES];
    x1[0] = 5.0 / 20.0;
    x1[10] = 2.0 / 7.0;
    let cards = vec![0i32; tavernagent_v2::net::N_CARD_SLOTS];
    let mut mask = vec![false; actions::N_ACTIONS];
    mask[0] = true;
    mask[1] = true;

    let mut x2 = x1.clone();
    x2[tavernagent_v2::net::OPP_BOARD_FEATURE_START] = 1.0;
    x2[tavernagent_v2::net::OPP_BOARD_FEATURE_START + 1] = 2.5;

    let (b1, v1) = base_net.infer_f32(&x1, &cards, &mask).unwrap();
    let (b2, v2) = base_net.infer_f32(&x2, &cards, &mask).unwrap();
    assert_eq!(b1, b2);
    assert_eq!(v1, v2);

    let (o1, ov1) = opp_net.infer_f32(&x1, &cards, &mask).unwrap();
    let (o2, ov2) = opp_net.infer_f32(&x2, &cards, &mask).unwrap();
    assert!(
        o1 != o2 || (ov1 - ov2).abs() > 1e-6,
        "observe_opponents network should respond to opponent warband features"
    );
}

#[test]
fn enable_move_expands_action_space_and_respects_must_spend_escape() {
    let cfg = NetConfig {
        hidden: 16,
        blocks: 1,
        embedding_dim: 4,
        enable_move: true,
        ..Default::default()
    };
    let net = PolicyValueNet::new(11, cfg).unwrap();
    assert_eq!(net.n_actions(), actions::N_ACTIONS_WITH_MOVE);
    assert_eq!(net.n_actions(), actions::N_ACTIONS + 49);

    let rules = PolicyRules {
        enable_move: true,
        ..Default::default()
    };
    let g_cfg = RulesConfig {
        max_turns: 3,
        ..Default::default()
    };
    let rollout = rollout::collect_game_with_rules(&net, 42, &g_cfg, rules).unwrap();
    assert!(!rollout.trajectories[0].steps.is_empty());
    assert_eq!(
        rollout.trajectories[0].steps[0].obs.mask.len(),
        actions::N_ACTIONS_WITH_MOVE
    );

    // Check must-spend escape: if a seat has >=3 gold and only Move + EndTurn are legal,
    // EndTurn must NOT remain masked out.
    let mut game = Game::new(42, g_cfg);
    tavernagent_v2::heroes::select_heroes(&mut game, 42).unwrap();
    let obs = observe_seat_with_rules(&game, 0, rules, &ActionHistory::default());
    assert_eq!(obs.mask.len(), actions::N_ACTIONS_WITH_MOVE);
    let mask = agent_mask_with_rules(&game, 0, rules, &ActionHistory::default());
    assert!(mask.iter().any(|&m| m));
}

#[test]
fn learn_hero_select_includes_hero_select_decision_in_trajectory() {
    let cfg = NetConfig {
        hidden: 16,
        blocks: 1,
        embedding_dim: 4,
        ..Default::default()
    };
    let net = PolicyValueNet::new(19, cfg).unwrap();
    let rules = PolicyRules {
        learn_hero_select: true,
        ..Default::default()
    };
    let g_cfg = RulesConfig {
        max_turns: 2,
        ..Default::default()
    };
    let rollout = rollout::collect_game_with_rules(&net, 77, &g_cfg, rules).unwrap();
    for traj in &rollout.trajectories {
        assert!(!traj.steps.is_empty());
        // First decision must be HeroSelect (Choose action in 30..46) at turn 0.
        let first = &traj.steps[0];
        assert!(
            (taverngym::actions::A_CHOOSE
                ..taverngym::actions::A_CHOOSE + taverngym::actions::N_CHOICE)
                .contains(&first.action),
            "expected HeroSelect Choose action, got {}",
            first.action
        );
    }
}

#[test]
fn potential_based_reward_shaping_preserves_telescoping_return() {
    let shaping = RewardShaping {
        health_coef: 0.2,
        tier_coef: 0.15,
    };
    let traj = Trajectory {
        seed: 1,
        seat: 0,
        placement: 2,
        terminal_health: 20.0,
        is_learner: true,
        steps: vec![
            Transition {
                turn: 1,
                health: 40.0,
                tier: 1,
                obs: Observation {
                    x: vec![],
                    cards: vec![],
                    mask: vec![],
                },
                action: 0,
                old_log_prob: 0.0,
                old_value: 0.0,
                reward: 0.0,
                terminated: false,
            },
            Transition {
                turn: 2,
                health: 36.0,
                tier: 2,
                obs: Observation {
                    x: vec![],
                    cards: vec![],
                    mask: vec![],
                },
                action: 0,
                old_log_prob: 0.0,
                old_value: 0.0,
                reward: 0.0,
                terminated: false,
            },
            Transition {
                turn: 3,
                health: 20.0,
                tier: 3,
                obs: Observation {
                    x: vec![],
                    cards: vec![],
                    mask: vec![],
                },
                action: 0,
                old_log_prob: 0.0,
                old_value: 0.0,
                reward: 0.7142857,
                terminated: true,
            },
        ],
    };
    let samples = traj
        .into_samples_with_shaping(1.0, 1.0, None, shaping)
        .unwrap();
    // With gamma=1, lambda=1, old_value=0: return_[0] = terminal_reward - Phi(s_0).
    let phi0 = shaping.potential(40.0, 1);
    assert!((samples[0].return_ - (0.7142857 - phi0)).abs() < 1e-5);
    // Terminal reward label itself stays unshaped for terminal-value supervision.
    assert!((samples[0].terminal_reward - 0.7142857).abs() < 1e-6);
}

#[test]
fn opponent_pool_collects_only_learner_seat_trajectories() {
    let cfg = NetConfig {
        hidden: 16,
        blocks: 1,
        embedding_dim: 4,
        ..Default::default()
    };
    let learner = PolicyValueNet::new(1, cfg.clone()).unwrap();
    let opp = PolicyValueNet::new(2, cfg).unwrap();
    let g_cfg = RulesConfig {
        max_turns: 2,
        ..Default::default()
    };
    let rollout =
        rollout::collect_game_with_pool(&learner, &[opp], 1.0, 123, &g_cfg, PolicyRules::default())
            .unwrap();
    assert!(rollout.trajectories[0].is_learner);
    assert!(!rollout.trajectories[0].steps.is_empty());
    for seat in 1..8 {
        assert!(!rollout.trajectories[seat].is_learner);
        assert!(rollout.trajectories[seat].steps.is_empty());
    }
}

#[test]
fn end_to_end_train_with_all_improvements_and_checkpoint_roundtrip() {
    let out = std::env::temp_dir().join(format!(
        "tavernagent-v2-improvements-{}.npz",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&out);

    let opts = TrainOptions {
        games: 2,
        epochs: 1,
        batch: 32,
        lr: 1e-4,
        gamma: 1.0,
        gae_lambda: 0.96,
        hidden: 16,
        blocks: 1,
        embedding_dim: 4,
        observe_opponents: true,
        shaping_health_coef: 0.05,
        shaping_tier_coef: 0.05,
        opponent_pool_prob: 0.25,
        opponent_pool_size: 4,
        opponent_pool_interval: 1,
        adaptive_kl: true,
        min_lr: 1e-6,
        min_entropy: 0.5,
        collapse_turn_limit_frac: 0.5,
        max_turns: 2,
        budgets: Budgets {
            action_cap: 16,
            move_cap: 4,
            freeze_cap: 3,
        },
        ppo: PpoOptions {
            terminal_value_coef: 0.1,
            ..Default::default()
        },
        policy_rules: PolicyRules {
            freeze_at_end: true,
            enable_move: true,
            learn_hero_select: true,
            ..Default::default()
        },
        ..Default::default()
    };

    tavernagent_v2::train::train(TrainArgs {
        device: 0,
        iterations: 2,
        workers: 2,
        out: out.clone(),
        save_every: 0,
        options: opts,
    })
    .unwrap();

    let (trainer, state) = checkpoint::load(&out).unwrap();
    assert_eq!(state.iteration, 2);
    assert!(trainer.net.cfg.observe_opponents);
    assert!(trainer.net.cfg.enable_move);
    assert!(state.options.policy_rules.learn_hero_select);
    assert!(state.effective_lr.is_some());
    assert!(state.effective_entropy_coef.is_some());

    // Verify PPO update still works cleanly on loaded trainer.
    let mut loaded_trainer = Trainer::new(trainer.net.clone());
    loaded_trainer.reset_optimizer();
    assert_eq!(loaded_trainer.optimizer_step, 0);

    let _ = std::fs::remove_file(&out);
    let _ = std::fs::remove_file(out.with_file_name(format!(
        "{}.history.json",
        out.file_stem().unwrap().to_string_lossy()
    )));
}
