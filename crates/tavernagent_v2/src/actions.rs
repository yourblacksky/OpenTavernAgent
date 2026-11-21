//! Compact agent actions. Engine actions and their indices remain unchanged.
use taverngym::actions as engine;
use taverngym::{Action, Game};

pub const N_POS: usize = engine::N_BOARD;
pub const N_ACTIONS: usize = engine::A_PLAY + engine::N_HAND * N_POS;
pub const A_MOVE: usize = N_ACTIONS;
pub const N_MOVE: usize = engine::N_BOARD * engine::N_BOARD;
pub const N_ACTIONS_WITH_MOVE: usize = N_ACTIONS + N_MOVE;

pub const fn n_actions(enable_move: bool) -> usize {
    if enable_move {
        N_ACTIONS_WITH_MOVE
    } else {
        N_ACTIONS
    }
}

pub const fn is_move(action: usize) -> bool {
    action >= A_MOVE && action < N_ACTIONS_WITH_MOVE
}

/// Resolve against the state *before* applying the action. A compatible occupied
/// position means Magnetize; an empty or incompatible position means normal Play.
pub fn engine_action(game: &Game, seat: usize, action: usize) -> Result<Action, &'static str> {
    engine_action_with(game, seat, action, false)
}

pub fn engine_action_with(
    game: &Game,
    seat: usize,
    action: usize,
    enable_move: bool,
) -> Result<Action, &'static str> {
    if action >= n_actions(enable_move) {
        return Err("bad agent action index");
    }
    if action >= A_MOVE {
        let k = action - A_MOVE;
        return Ok(Action::Move {
            from: k / N_POS,
            to: k % N_POS,
        });
    }
    if action < engine::A_PLAY {
        return Action::decode(action).ok_or("bad agent action index");
    }
    let k = action - engine::A_PLAY;
    let (hand, pos) = (k / N_POS, k % N_POS);
    Ok(if game.can_magnetize(seat, hand, pos) {
        Action::Magnetize { hand, board: pos }
    } else {
        Action::Play { hand, pos }
    })
}

pub fn legal_mask(game: &Game, seat: usize) -> Vec<bool> {
    legal_mask_with(game, seat, false)
}

pub fn legal_mask_with(game: &Game, seat: usize, enable_move: bool) -> Vec<bool> {
    let mask = game.legal_mask(seat);
    (0..n_actions(enable_move))
        .map(|a| {
            mask[engine_action_with(game, seat, a, enable_move)
                .unwrap()
                .encode()]
        })
        .collect()
}

pub fn apply(game: &mut Game, seat: usize, action: usize) -> Result<(), &'static str> {
    let action = engine_action(game, seat, action)?;
    game.apply(seat, action)
}

/// Policy semantics belong to the checkpoint; the simulator action space is unchanged.
pub fn apply_with_rules(
    game: &mut Game,
    seat: usize,
    action: usize,
    rules: crate::rules::PolicyRules,
) -> Result<(), &'static str> {
    apply_with(game, seat, action, rules, |game, seat, action| {
        game.apply(seat, action)
    })
}

/// Shared action expansion for ordinary evaluation and observational replay hooks.
pub(crate) fn apply_with(
    game: &mut Game,
    seat: usize,
    action: usize,
    rules: crate::rules::PolicyRules,
    mut apply: impl FnMut(&mut Game, usize, Action) -> Result<(), &'static str>,
) -> Result<(), &'static str> {
    if rules.freeze_at_end && action == engine::A_FREEZE {
        let legal = game.legal_mask(seat);
        if !legal[engine::A_END] || !legal[engine::A_FREEZE] {
            return Err("freeze-and-end is not legal");
        }
        // An already frozen shop need not be toggled off. Partially frozen shops
        // become fully frozen, matching the engine Freeze button.
        if !game.players[seat].shop.iter().all(|slot| slot.frozen) {
            apply(game, seat, Action::Freeze)?;
        }
        apply(game, seat, Action::EndTurn)
    } else {
        let engine = engine_action_with(game, seat, action, rules.enable_move)?;
        apply(game, seat, engine)
    }
}
