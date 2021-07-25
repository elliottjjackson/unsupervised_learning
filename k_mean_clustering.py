import numpy as np
import matplotlib.pyplot as plt

class KMeanCluster:
    def __init__(self,seed,datapoints_num,cluster_num):
        """Initialises random dataset using a seed number (seed) and
        guesses a defined number of cluster means (cluster_num).
        Returns a plot of the dataset and initial cluster mean guesses"""
        self.seed = seed
        self.datapoints_num = datapoints_num
        self.cluster_num = cluster_num
        np.random.seed(self.seed)
        self.cluster_coord = np.random.rand(self.datapoints_num,2)*100
        half_points = int(self.datapoints_num / 2)
        self.cluster_coord[:half_points,:] = self.cluster_coord[:half_points,:] + 100
        self.cluster_coord = np.floor(self.cluster_coord)
        cluster_mean = np.random.rand(self.cluster_num,2)*200
        cluster_mean = np.floor(cluster_mean)
        self.cluster_mean = cluster_mean
        self.plot_original_dataset()

    def classify_cluster(self):
        """Classifies the cluster points depending on their euclidian distance
        from the closest mean point"""
        for n in range(self.cluster_num):
            difference_matrix = (self.cluster_coord - self.cluster_mean[n])**2
            difference_matrix = np.sum(difference_matrix,axis=1)
            difference_matrix = np.sqrt(difference_matrix)
            difference_matrix = np.reshape(difference_matrix,(self.datapoints_num,-1))
            if n == 0:
                result_diff_matrix = difference_matrix
            else:
                result_diff_matrix = np.append(result_diff_matrix,difference_matrix,axis=1)
        min_diff_matrix = np.amin(result_diff_matrix,axis=1)
        min_diff_matrix = np.array(min_diff_matrix)[:,None]
        cluster_allocation_filter = np.equal(result_diff_matrix,min_diff_matrix)
        self.cluster_classified = []
        for cols in range(self.cluster_num):
            filtered_cluster_column = self.cluster_coord[cluster_allocation_filter[:,cols]]
            if filtered_cluster_column.size == 0:
                self.cluster_num -= 1 # Removes a mean point if there are no points classified to it.
            else:
                self.cluster_classified.append(filtered_cluster_column)
        self.plot_result_dataset()

    def next_classify_iteration(self):
        """Redefines the mean points as the mean of their classified
        dataset, then runs the classification process again."""
        new_means = []
        for n in range(self.cluster_num):
            x = np.mean(self.cluster_classified[n][:,0])
            y = np.mean(self.cluster_classified[n][:,1])
            new_means.append([x,y])
        self.cluster_mean = new_means
        self.classify_cluster()

    def __plot_set_up(self):
        """Sets up the matlib.plt for all plotting functions"""
        self.color_table = ['b','g','r','c','m','y','k']
        fig=plt.figure()
        self.ax=fig.add_axes([0,0,1,1])
        self.ax.set_title('scatter plot')
        self.ax.set_xlabel('X Values')
        self.ax.set_ylabel('Y Values')
        plt.xlim((0,200))
        plt.ylim((0,200))

    def plot_original_dataset(self):
        """Plots the starting dataset and mean guesses prior to classification"""
        self.__plot_set_up()
        self.ax.scatter(self.cluster_coord[:,0],self.cluster_coord[:,1],color=self.color_table[0])
        for num in range(self.cluster_num):
            self.ax.scatter(self.cluster_mean[num][0],self.cluster_mean[num][1],marker="x",color=self.color_table[num%7])
        plt.show()

    def plot_result_dataset(self):
        """Plots the classified dataset based on the mean guesses"""
        self.__plot_set_up()
        for num in range(self.cluster_num):
            self.ax.scatter(np.array(self.cluster_classified[num])[:,0],np.array(self.cluster_classified[num])[:,1],color=self.color_table[num%7])
            self.ax.scatter(self.cluster_mean[num][0],self.cluster_mean[num][1],marker="x",color=self.color_table[num%7])
        plt.show()

cluster1 = KMeanCluster(1,100,3)
cluster1.classify_cluster()
cluster1.next_classify_iteration()
cluster1.next_classify_iteration()
cluster1.next_classify_iteration()
cluster1.next_classify_iteration()