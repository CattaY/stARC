import numpy as np
from scipy.stats import *
import os

def sprandasym(n, density): ###生成随机稀疏矩阵（有向）, r_type == 2
    rvs = stats.uniform(loc=-1, scale=2).rvs
    X = sparse.random(n, n, density=density, data_rvs=rvs)
    X = X.todense()
    return X

def sprandsym(n, density): ###生成随机对称稀疏矩阵, r_type == 1
    rvs = stats.uniform().rvs
    X = sparse.random(n, n, density=density, data_rvs=rvs)
    result = sparse.triu(X) + sparse.triu(X).T - sparse.diags(X.diagonal())
    result = result.todense()
    return result

# ----------------------------------------------------------------------
# Main Execution (for generating random W_r of reservoir network)
# ----------------------------------------------------------------------
if __name__ == "__main__":
  runs = 200  # generated numbers
  a = 5  # scale parameter
  n = int(200) # reservoir nodes
  con = 0.2 # average connectivity
  k = int(n*con) # average degree
  r_type = 1
  path = "../Data/Wr_groups/k=%.2f/%in%i" % (con, k, n)
      if not os.path.exists(path):
          os.makedirs(path)

  for i in range(runs):
      if r_type == 1:
          W_r = sprandsym(n, con)
      elif r_type == 2:
          W_r = a * (2 * sprandasym(n, con) - 1)
      else:
          print("r type error\n")
          break
      eig = np.max(abs(np.linalg.eig(W_r)[0]))
      W_r = (1/eig)*W_r
      np.savetxt(r"../Data/Wr_groups/k=%.2f/%in%i/Wr_%in%i_a%i_%i.txt"
                 % (con, k, n, k, n, a, i+1), W_r, fmt='%f', delimiter=',')
      print("%i-th W_r generated successfully.\t##%in%i" % (i+1, k, n))
