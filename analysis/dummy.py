master_seed = 0
sim.n_vp = 8

sli_str = r"0 << \n"
sli_str += r"%/rng "
sli_str += "%i [0 %i 1 sub] "%(master_seed, sim.n_vp)
sli_str += r"add Range { rngdict/MT19937 :: exch CreateRNG } Map  % local RNG, seeded \n"
sli_str += r"%/grng rngdict/MT19937 :: "
sli_str += "%i %i add CreateRNG "%(master_seed, sim.n_vp)
sli_str += "% global RNG \n"
sli_str += "/rng_seeds %i [0 %i 1 sub] add Range "%(master_seed, sim.n_vp)
sli_str += "% local RNG seeds \n"
sli_str += "/grng_seed %i %i add "%(master_seed, sim.n_vp)
sli_str += "% global RNG seed \n"
sli_str += ">> SetStatus"
