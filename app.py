from flask import Flask, send_from_directory, jsonify
import os
from main import get_user_info

app = Flask("github_graph")

# Function that returns user information as a dictionary
# def get_user_info(user):
#     user_info = {
#         'name': 'John Doe',
#         'age': 30,
#         'email': 'john.doe@example.com'
#     }
#     # Assume this function fetches user info from a database or other source based on the provided username
#     return user_info

@app.route('/')
def index():
    return send_from_directory('./static', 'index.html')

@app.route('/<path:path>')
def serve_static_files(path):
    return send_from_directory('./static', path)

@app.route('/api/user/<string:user>', methods=['GET'])
def user_info(user):
    user_info_dict = get_user_info(user)
    return jsonify(user_info_dict)

if __name__ == '__main__':
    import sys
    print("use r for release")
    if sys.argv[-1] == 'r':
      app.run(debug=False, host="0.0.0.0")
    else:
      app.run(debug=True)