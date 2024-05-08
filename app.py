import os
import tempfile
import shutil
from flask import Flask, request, jsonify
import git
import subprocess
import requests
import openai

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.json
        event_type = request.headers.get('X-GitHub-Event', 'push')

        if 'zen' in data:
            return jsonify({'msg': 'ping通ってよかったね'}), 200

        if event_type == 'push':
            if 'after' not in data:
                return jsonify({'error': "Missing 'after' key in request data."}), 400

            with tempfile.TemporaryDirectory(prefix='auto-ai-review-') as temp_dir:
                repo = git.Repo.clone_from(data['repository']['clone_url'], temp_dir)
                origin = repo.remotes.origin
                
                origin.fetch()
                repo.git.checkout('main')
                repo.git.reset('--hard', 'origin/main')
                
                commit_sha = data['after']
                review_branch = 'auto-ai-review'
                
                if review_branch not in repo.branches:
                    repo.git.branch(review_branch)
                repo.git.checkout(review_branch)
                
                changed_files = [item.a_path for item in repo.index.diff(None)]
                best_optimizations = {}

                for file_path in changed_files:
                    full_path = os.path.join(temp_dir, file_path)
                    optimized_code = optimize_code_with_ai(full_path)
                    
                    with open(full_path, 'w') as file:
                        file.write(optimized_code)
                    
                    result = subprocess.run(["pytest", full_path], capture_output=True, text=True)
                    if result.returncode == 0:
                        repo.git.add(file_path)
                        repo.git.commit('-m', f'Optimize {file_path}')

                if best_optimizations:
                    origin.push(review_branch)
                    create_pull_request(os.getenv('GITHUB_TOKEN'), data['repository']['full_name'], 'Automated Pull Request for Code Optimization', 'This pull request contains automated code optimizations.', review_branch, 'main')

            return 'Webhook received and processed successfully!', 200

        elif event_type == 'status':
            return jsonify({'msg': 'Status event received'}), 200

        else:
            return jsonify({'error': 'Unsupported event type'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def optimize_code_with_ai(file_path):
    openai.api_key = os.getenv('OPENAI_API_KEY')
    with open(file_path, 'r') as file:
        original_code = file.read()

    response = openai.Completion.create(
        engine="gpt-4-turbo",
        prompt=f"### Python\n# Original Code\n{original_code}\n# Optimize the code above for better performance and readability.",
        max_tokens=150
    )
    return response.choices[0].text.strip()

def create_pull_request(token, repo, title, body, head, base):
    url = f"https://api.github.com/repos/{repo}/pulls"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    data = {"title": title, "body": body, "head": head, "base": base}
    response = requests.post(url, headers=headers, json=data)
    return response.json()

if __name__ == '__main__':
    app.run(port=5000, debug=True)