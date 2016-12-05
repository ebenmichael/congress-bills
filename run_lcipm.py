from lcipm.models import lcipm
from lcipm.models import VB
import sys
import numpy as np
import os


def run_lcipm(interactions, caucus, dim, n_communities,
              var_x, outdir, max_laps=30):
    """Run LCIPM with given parameters
    Args:
        interactions: ndarray shape (n_interactions, 3), interaction data
        caucus: ndarray shape(n_user, n_user), caucus co-membership data
        dim: int, dimension of ideal points
        n_communities, int, number of latent communities
        var_x: float, prior variance on ideal points
        outidr: string, directory to save to
        max_laps: int, max number of laps
    Return:
       lc: LCIPM, trained LCIPM model
    """
    lc = lcipm.LCIPM(dim, n_communities=n_communities,
                     ip_prior_var=var_x)
    # fit with variational inference
    vb = VB.VB(maxLaps=max_laps)
    elbo, _ = vb.run(lc, (interactions, caucus), save=True, outdir=outdir)
    return(lc, elbo)


if __name__ == "__main__":
    if len(sys.argv) != 8:
        print("Usage: python run_lcipm.py data/combined_data/interactions.dat "
              + "data/combined_data/membership.dat dim K var_x max_laps test")
    else:
        interactions = np.loadtxt(sys.argv[1])
        caucus = np.loadtxt(sys.argv[2])
        caucus = np.dot(caucus, caucus.T)
        dim = int(sys.argv[3])
        n_communities = int(sys.argv[4])
        var_x = float(sys.argv[5])
        max_laps = int(sys.argv[6])
        outdir = sys.argv[7]

        lc, elbo = run_lcipm(interactions, caucus, dim, n_communities,
                             var_x, outdir, max_laps=max_laps)
        # save lc
        lc.save(os.path.join(outdir, "lcipm_" + str(dim) + "_"
                             + str(n_communities) + "_" + str(var_x)))
        # save elbo
        np.savetxt(os.path.join(outdir, "elbo.dat"), np.array(elbo))
