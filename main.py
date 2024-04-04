import requests
import queue
from time import sleep
import os
import json

session = requests.Session()
personal_access_token = open(".ghtoken").read()
session.headers.update({'Authorization': f'token {personal_access_token}'})

def get_users(username, user_type, max_users_count=None):
    users = []
    url = f"https://api.github.com/users/{username}/{user_type}"

    while url:
        response = session.get(url)
        if response.status_code == 200:
            response_data = response.json()
            for user in response_data:
              if max_users_count and len(users) >= max_users_count:
                break
              users.append(user['login'])

            if 'next' in response.links and (max_users_count is None or len(users) < max_users_count):
                url = response.links['next']['url']
            else:
                url = None
        else:
            print(f"Failed to get {user_type}, status code: {response.status_code}")
            url = None

    return users

def get_user_info(username):
  cache_file_path = f"cache/{username}"
  if os.path.exists(cache_file_path):
    with open(cache_file_path, 'r') as file:
      try:
        return json.load(file)
      except:
        print("failed to load cache for", username)

  user = {
    'username': username,
  }

  url = f'https://api.github.com/users/{username}'
  response = session.get(url)
  if response.status_code == 200:
    response_data = response.json()
    user['followers_cnt'] = response_data['followers']
    user['following_cnt'] = response_data['following']
  else:
    print(f"Failed to get {username}, status code: {response.status_code}")
    return None

  user['followers'] = get_users(username, 'followers', max_users_count=10)
  user['following'] = get_users(username, 'following', max_users_count=10)
  with open(cache_file_path, 'w') as file:
      json.dump(user, file)
  return user


# def bfs(username):
#   checked = set()

#   q = queue.Queue()
#   q.put(username)
#   while not q.empty():
#     name = q.get()
#     if name in checked:
#       continue
#     checked.add(name)
#     user = get_user_info(name)
#     print(user)
#     if user['followers_cnt'] < 10:
#       for follower in user['followers']:
#           q.put(follower)
#     if user['following_cnt'] < 10:
#       for following in user['following']:
#         q.put(following)
#     sleep(0.5)

# if __name__ == "__main__":
#   username = 'misha1sh'
#   # print(get_user_info(username))
#   print(bfs(username))
