# import numpy as np

# def agglomerative_clustering(X, n_clusters=2):
#     """
#     AGGLOMERATIVE = Start ALONE, MERGE friends
#     X: Your numbers (like toy sizes)
#     n_clusters: How many groups at the end
#     """
    
#     # STEP 1: Start with each number in its OWN group
#     # Convert 1D list to 2D array for distance calculations
#     X_array = np.array(X).reshape(-1, 1)  # Make it 2D: [[9], [8], [7], ...]
#     clusters = [[i] for i in range(len(X_array))]
#     # Example: X=[9,8,7,6,6,6] → clusters = [[0], [1], [2], [3], [4], [5]]
    
#     print("🎮 START: Each number is ALONE!")
#     print(f"Clusters: {clusters}")
    
#     # Keep merging until we have n_clusters
#     while len(clusters) > n_clusters:
        
#         # STEP 2: Find which two groups are CLOSEST
#         min_distance = float('inf')  # Start with INFINITE distance
#         best_pair = (0, 1)  # Which groups to merge
        
#         # Look at EVERY possible pair of groups
#         for i in range(len(clusters)):
#             for j in range(i + 1, len(clusters)):
                
#                 # Find the CLOSEST numbers between these two groups
#                 smallest_dist = float('inf')
                
#                 # Check every number in group i with every number in group j
#                 for idx1 in clusters[i]:
#                     for idx2 in clusters[j]:
#                         # Calculate distance (absolute difference for 1D)
#                         dist = abs(X_array[idx1][0] - X_array[idx2][0])
#                         if dist < smallest_dist:
#                             smallest_dist = dist
                
#                 # Is this the CLOSEST pair we've seen?
#                 if smallest_dist < min_distance:
#                     min_distance = smallest_dist
#                     best_pair = (i, j)
        
#         # STEP 3: Merge the BEST FRIEND groups!
#         i, j = best_pair
#         print(f"\n🤝 MERGING: Group {clusters[i]} and Group {clusters[j]}")
#         print(f"   Why? They're only {min_distance} apart!")
        
#         # Move all numbers from group j into group i
#         clusters[i] = clusters[i] + clusters[j]
#         # Remove the now-empty group j
#         clusters.pop(j)
        
#         print(f"   New clusters: {clusters}")
    
#     # Return the final groups (as actual values, not indices)
#     print("\n🎉 FINAL GROUPS:")
#     result = []
#     for group in clusters:
#         values = [X[idx] for idx in group]
#         result.append(values)
#         print(f"   Group: {values}")
    
#     return result

# # Let's test with your data!
# X = [9, 8, 7, 6, 6, 6]
# print("="*50)
# print("AGGLOMERATIVE CLUSTERING (Friend Finder)")
# print("="*50)
# print(f"Data: {X}")
# result = agglomerative_clustering(X, n_clusters=2)
import numpy as np

def divisive_clustering(X, n_clusters=2):
    """
    X: List of numbers to cluster (e.g., chocolate sizes)
    n_clusters: How many groups you want
    Returns: List of lists (each sublist is a cluster)
    """
    
    # Step 1: Start with ALL points in ONE cluster
    # Create list of indexes: [0, 1, 2, ..., len(X)-1]
    clusters = [list(range(len(X)))]
    
    # Step 2: Keep splitting until we have enough clusters
    while len(clusters) < n_clusters:
        
        # Step 3: Find which cluster to split
        max_variance = -1
        split_idx = 0
        
        for i, cluster in enumerate(clusters):
            # Only consider clusters with >1 point
            if len(cluster) > 1:
                # Get actual values for this cluster
                cluster_values = [X[idx] for idx in cluster]
                # Calculate how "spread out" they are
                current_variance = np.var(cluster_values)
                
                # Track the cluster with maximum variance
                if current_variance > max_variance:
                    max_variance = current_variance
                    split_idx = i
        
        # Step 4: Remove the chosen cluster
        cluster_to_split = clusters.pop(split_idx)
        
        # Step 5: Split at the median
        values = [X[idx] for idx in cluster_to_split]
        median_val = np.median(values)
        
        # Create two new clusters
        left_cluster = []
        right_cluster = []
        
        for idx in cluster_to_split:
            if X[idx] <= median_val:
                left_cluster.append(idx)
            else:
                right_cluster.append(idx)
        
        # Step 6: Add the two new clusters
        clusters.append(left_cluster)
        clusters.append(right_cluster)
    
    # Step 7: Convert indexes back to values
    final_clusters = []
    for cluster in clusters:
        cluster_values = [X[idx] for idx in cluster]
        final_clusters.append(cluster_values)
    
    return final_clusters

# Test it!
chocolate_sizes = [9, 8, 7, 6, 6, 6]
print("Chocolate sizes:", chocolate_sizes)
groups = divisive_clustering(chocolate_sizes, 2)
print("After divisive clustering:", groups)