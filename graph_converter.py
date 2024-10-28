
import pymongo

mongodb_pass = open(".mongodb_reader").read()
client = pymongo.MongoClient(f'mongodb://ghresearch:{mongodb_pass}@localhost:27017/')

db = client['github_db']
collection = db['users']
edges = set()
visited_nodes = set()

for user in collection.find():
    username = user['username']
    visited_nodes.add(username)

    # Process 'following' relationships
    following = user.get('following', [])
    for followed_user in following:
        edges.add((username, followed_user))

    # Process 'followers' relationships
    followers = user.get('followers', [])
    for follower in followers:
        edges.add((follower, username))

output_file = 'edges.txt'
with open(output_file, 'w') as f:
    for edge in edges:
        f.write(f"{edge[0]} {edge[1]}\n")

output_file = 'edges_unoriented.txt'
with open(output_file, 'w') as f:
    for edge in edges:
      if (edge[1], edge[0]) in edges:
        f.write(f"{edge[0]} {edge[1]}\n")


output_file = 'edges_visited.txt'
with open(output_file, 'w') as f:
    for edge in edges:
      if edge[0] in visited_nodes and edge[1] in visited_nodes:
        f.write(f"{edge[0]} {edge[1]}\n")



print(f"Edges have been written to {output_file}")
