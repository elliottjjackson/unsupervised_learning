import numpy as np
import matplotlib.pyplot as plt

class KMeanCluster:
    def __init__(self,seed,datapoints_num,cluster_num):
        self.seed = seed
        self.datapoints_num = datapoints_num
        self.cluster_num = cluster_num
        np.random.seed(self.seed)
        self.cluster_coord = np.random.rand(self.datapoints_num,2)*100
        half_points = self.datapoints_num / 2
        self.cluster_coord[:10,:] = self.cluster_coord[:10,:] + 100
        self.cluster_coord = np.floor(self.cluster_coord)
        cluster_mean = np.random.rand(self.cluster_num,2)*200
        cluster_mean = np.floor(cluster_mean)
        self.cluster_mean = cluster_mean


    def next_classify_iteration(self):
        pass

    def classify_cluster(self):
            for n in range(self.cluster_num):
                difference_matrix = (self.cluster_coord - self.cluster_mean[n])**2
                difference_matrix = np.sum(difference_matrix,axis=1)
                difference_matrix = np.sqrt(difference_matrix)
                difference_matrix = np.reshape(difference_matrix,(self.datapoints_num,-1))
                if n == 0:
                    result_diff_matrix = difference_matrix
                else:
                    result_diff_matrix = np.append(result_diff_matrix,difference_matrix,axis=1)
            min_diff_matrix = np.amax(result_diff_matrix,axis=1)
            min_diff_matrix = np.array(min_diff_matrix)[:,None]
            # print(self.cluster_coord,"\n",self.cluster_mean,"\n",difference_matrix,"\n",result_diff_matrix,"\n",min_diff_matrix)
            cluster_allocation_filter = np.equal(result_diff_matrix,min_diff_matrix)
            self.cluster_classified = []
            for cols in range(self.cluster_num):
                self.cluster_classified.append(self.cluster_coord[cluster_allocation_filter[:,cols]])

    def plot_dataset(self):
        color_table = ['b','g','r','c','m','y','k']
        fig=plt.figure()
        ax=fig.add_axes([0,0,1,1])
        ax.set_title('scatter plot')
        ax.set_xlabel('X Values')
        ax.set_ylabel('Y Values')
        for num in range(self.cluster_num):
            ax.scatter(np.array(self.cluster_classified[num])[:,0],np.array(self.cluster_classified[num])[:,1],color=color_table[num%7])
            ax.scatter(self.cluster_mean[num][0],self.cluster_mean[num][1],marker="x",color=color_table[num%7])
        plt.xlim((0,200))
        plt.ylim((0,200))
        plt.show()
        print(self.cluster_mean[1][0], self.cluster_mean[1][1])   

cluster1 = KMeanCluster(10,20,2)
cluster1.classify_cluster()
cluster1.plot_dataset()
# cluster1.plot_dataset()