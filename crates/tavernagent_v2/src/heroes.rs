//! Hero selection is an external policy, separate from recruit-phase learning.

use rand::SeedableRng;
use rand::seq::IndexedRandom;
use rand_pcg::Pcg64Mcg;
use taverngym::Game;
use taverngym::actions::{A_CHOOSE, N_CHOICE};

use taverngym::engine::Phase;

/// Uniformly choose an offered hero for every seat, reproducibly from `seed`.
///
/// No network calls or training samples. A live hero-ranking policy belongs at this
/// boundary. Recruit-phase Discovers opened by hero effects remain for the normal policy.
pub fn select_heroes(game: &mut Game, seed: u64) -> Result<(), String> {
    select_heroes_where(game, seed, |_| true)
}

/// Uniformly choose an offered hero only for seats where `use_uniform(seat)` is true.
/// Seats where `use_uniform(seat)` is false remain in `Phase::HeroSelect` so the policy
/// can choose their hero via `Action::Choose`.
pub fn select_heroes_where(
    game: &mut Game,
    seed: u64,
    mut use_uniform: impl FnMut(usize) -> bool,
) -> Result<(), String> {
    if game.phase != Phase::HeroSelect {
        return Ok(());
    }
    let mut rng = Pcg64Mcg::seed_from_u64(seed ^ 0x4845_524F);
    for seat in 0..game.players.len() {
        if !use_uniform(seat) {
            continue;
        }
        if game.needs_action(seat) {
            let mask = game.legal_mask(seat);
            let choices: Vec<usize> = (A_CHOOSE..A_CHOOSE + N_CHOICE)
                .filter(|&a| mask[a])
                .collect();
            let action = *choices
                .choose(&mut rng)
                .ok_or("hero selection has no legal offered hero")?;
            game.apply_index(seat, action).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}
