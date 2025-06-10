use rand::Rng;

pub trait TSProblem {
    type Individ: Clone;
    type Move: Clone + PartialEq;

    fn random_individual<R: Rng>(rng: &mut R) -> Self::Individ;

    fn fitness(ind: &Self::Individ) -> f64;

    fn neighbourhood<R: Rng>(
        rng: &mut R,
        ind: &Self::Individ,
        size: usize,
    ) -> Vec<(Self::Individ, Self::Move)>;

    fn apply_move(ind: &mut Self::Individ, mv: &Self::Move);

    fn repair(ind: &mut Self::Individ) {
        let _ = ind;
    }
}
