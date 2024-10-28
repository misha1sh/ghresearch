import requests
import queue
import time
import os
import json

session = requests.Session()
personal_access_token = open(".ghtoken").read()
session.headers.update({'Authorization': f'token {personal_access_token}'})

mongodb_pass = open(".mongodb_admin").read()


import pymongo
client = pymongo.MongoClient(f'mongodb://admin:{mongodb_pass}@localhost:27017/')

db = client['github_db']
collection = db['users']
collection.create_index([("username", pymongo.ASCENDING)], unique=True)


def get_users(username, user_type, max_users_count=None):
    users = []
    url = f"https://api.github.com/users/{username}/{user_type}"

    while url:
        response = session.get(url)
        time.sleep(0.5)

        try:

          remaining_limit = response.headers['X-RateLimit-Remaining']
          if int(remaining_limit) <= 10:
            sleep_time = float(response.headers['X-RateLimit-Reset']) - time.time() + 10.
            print('Waiting for limit to recover', sleep_time, 'seconds')
            time.sleep(sleep_time)

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
              print(f"Failed to get {user_type}, status code: {response.status_code}, text: {response.text}, headers: {response.headers}")
              url = None
        except:
          print(response.text)
          raise

    return users

def get_user_info(username):
  user = {
    'username': username,
  }
  user_in_db = collection.find_one(user)
  if user_in_db and ('followers' not in user_in_db or 'following' not in user_in_db):
    collection.delete_one(user_in_db)
    print('deleted', user_in_db)
    user_in_db = None

  if user_in_db:
    return user_in_db

  url = f'https://api.github.com/users/{username}'
  response = session.get(url)
  if response.status_code == 200:
    response_data = response.json()
    user['followers_cnt'] = response_data['followers']
    user['following_cnt'] = response_data['following']
  else:
    print(f"Failed to get {username}, status code: {response.status_code}")
    return None

  if user['followers_cnt'] <= 200:
    user['followers'] = get_users(username, 'followers', max_users_count=200)
  else:
    user['followers'] = []

  if user['following_cnt'] <= 200:
    user['following'] = get_users(username, 'following', max_users_count=200)
  else:
    user['following'] = []

  collection.insert_one(user)
  return user


from queue import Queue
visited = set()
users_queue = Queue()
users_queue.put('misha1sh')
while not users_queue.empty():
  username = users_queue.get()
  user = get_user_info(username)
  if 'followers' not in user:
    print(user)

  for new_user in (user['followers'] + user['following']):
    if new_user not in visited:
      users_queue.put(new_user)
      visited.add(new_user)
  print("Total count: ", collection.count_documents({}))
