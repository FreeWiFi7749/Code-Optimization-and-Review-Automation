from dotenv import load_dotenv
load_dotenv()
import os
import tempfile
import shutil
from flask import Flask, request, jsonify
import git
import subprocess
import requests
import openai
from openai import OpenAI
import logging
import threading
import sys

app = Flask(__name__)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handle_push_event(data):
    try:
        commit_sha = data.get('sha', data.get('head_commit', {}).get('id'))
        logger.info(f'Commit SHA: {commit_sha}')
        if not commit_sha:
            return jsonify({'error': "Missing 'sha' or 'head_commit.id' key in request data."}), 400

        with tempfile.TemporaryDirectory(prefix='auto-ai-review-') as temp_dir:
            repo = git.Repo.clone_from(data['repository']['clone_url'], temp_dir)
            origin = repo.remotes.origin
            logger.info(f'Cloned repository: {data["repository"]["full_name"]}')
            
            repo.git.fetch()
            repo.git.checkout(data.get('ref', '').split('/')[-1])
            logger.info(f'Checked out branch: {data.get("ref", "").split("/")[-1]}')
            
            changed_files = repo.git.diff('HEAD~1', name_only=True).split()
            logger.info(f'Changed files: {changed_files}')

            review_branch = 'auto-ai-review'
            try:
            
                repo.remotes.origin.fetch()
                if review_branch not in repo.branches:
                    repo.git.branch(review_branch)
                    logger.info(f'Created branch: {review_branch}')
                repo.git.checkout(review_branch)
                logger.info(f'Checked out branch: {review_branch}')
            except Exception as e:
                logger.error(f'Failed to handle branch operations: {str(e)}', exc_info=True)
                return jsonify({'error': f'Failed to handle branch operations: {str(e)}'}), 500
            
            if not changed_files:
                logger.info('No files have been changed.')
                return jsonify({'msg': 'No files have been changed'}), 200

            best_optimizations = {}

            for file_path in changed_files:
                full_path = os.path.join(temp_dir, file_path)
                logger.info(f'Original code for {file_path}: {open(full_path, "r").read()}')
                optimized_code = optimize_code_with_ai(full_path)
                logger.info(f'Optimized code for {file_path}: {optimized_code}')

                if optimized_code:
                    with open(full_path, 'w') as file:
                        file.write(optimized_code)
                    logger.info(f'Updated code for {file_path}: {open(full_path, "r").read()}')

                    repo.git.add(file_path)
                    status = repo.git.status()
                    logger.info(f'Git status after add: {status}')
                    if repo.is_dirty():
                        repo.git.commit('-m', f'Optimize {file_path}')
                        logger.info(f'Committed changes to {file_path} on {review_branch}')
                    else:
                        logger.info('No changes detected, skipping commit.')

                    result = subprocess.run(["pytest", full_path], capture_output=True, text=True)
                    logger.info(f'Pytest output for {file_path}: {result.stdout}')
                    if result.returncode == 0:
                        repo.git.add(file_path)
                        repo.git.commit('-m', f'Optimize {file_path}')
                        best_optimizations[file_path] = optimized_code
                    else:
                        logger.info(f'Pytest failed for {file_path}: {result.stderr}')
                        again_optimized_code = again_optimize_code_with_ai(full_path)
                        if again_optimized_code:
                            with open(full_path, 'w') as file:
                                file.write(again_optimized_code)
                            retest_result = subprocess.run(["pytest", full_path], capture_output=True, text=True)
                            if retest_result.returncode == 0:
                                repo.git.add(file_path)
                                repo.git.commit('-m', f'Re-optimized {file_path} after fixing errors')
                                best_optimizations[file_path] = again_optimized_code
                            else:
                                logger.info(f'Retest failed for {file_path}: {retest_result.stderr}')
                                repo.git.add(file_path)
                                repo.git.commit('-m', f'Fix errors in {file_path}')
                                best_optimizations[file_path] = again_optimized_code
                                logger.info(f'Fix errors in {file_path}')

            if best_optimizations:
                for file_path, optimized_code in best_optimizations.items():
                    with open(os.path.join(temp_dir, file_path), 'w') as file:
                        file.write(optimized_code)
                    repo.git.add(file_path)
                    repo.git.commit('-m', f'Optimize {file_path}')
                    logger.info(f'Optimized code for {file_path}: {optimized_code}')
                origin.push(review_branch)
                logger.info(f'Pushed {review_branch} to remote')
                create_pull_request(os.getenv('GITHUB_TOKEN'), data['repository']['full_name'], 'Automated Pull Request for Code Optimization', 'This pull request contains automated code optimizations.', review_branch, 'main')
                logger.info('Pull request created.')
            else:
                logger.info('No optimizations found.')
    except Exception as e:
        with app.app_context():
            logger.error(f'エラーが発生しました: {str(e)}', exc_info=True)
            return jsonify({'error': str(e)}), 500

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    event_type = request.headers.get('X-GitHub-Event', 'push')

    if event_type == 'push':
        with app.app_context():
            thread = threading.Thread(target=handle_push_event, args=(data,))
            thread.start()
        return jsonify({'msg': 'Webhook received and processing in background'}), 200

    elif event_type == 'status':
        logger.info(f'Status event received: {data}')
        return jsonify({'msg': 'Status event received'}), 200

    else:
        logger.error(f'Unsupported event type: {event_type}')
        return jsonify({'error': 'Unsupported event type'}), 400

def optimize_code_with_ai(file_path):
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error('OPENAI_API_KEY 環境変数が設定されていません。')
        return None

    client = OpenAI(api_key=api_key)
    
    with open(file_path, 'r') as file:
        original_code = file.read()
        print(f'Optimizing code: {original_code}')

    prompt = f"""
    ### Python
    # Original Code
    {original_code}
    # Instructions
    - コードのパフォーマンスを向上させる
    - コードの可読性を高めるためにコメントを追加する
    - 不要なコードや冗長な部分を削除する
    - コードの構造を改善する
    - コードの品質を向上させる
    - コードのテストを実行して問題がないことを確認する
    - コードの最適化を行う
    - コードのエラーを修正する
    - コードのセキュリティを向上させる
    - コードの可用性を向上させる
    - コードの保守性を向上させる
    - コードの拡張性を向上させる
    - コードの再利用性を向上させる
    - コードの効率性を向上させる
    - コードの信頼性を向上させ
    - コードの柔軟を向上させる
    - コードの安全性を向上させる
    - コードのテストカバレッジを向上させる
    - コードのリファクタリングを行う
    - コードのパフォーマンスを測定する
    - コードの品質を評価する
    - コードの可読性を評価する
    - コードのセキュリティを評価する
    - 「#」を使ってコメントを追加する
    - 「""""""」は使用しない
    # Optimize the code above according to the instructions.
    """
    best_code = None
    best_score = float('-inf')

    messages = [
        {"role": "system", "content": prompt}
    ]

    for _ in range(4):
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.5
        )
        logger.info(f'OpenAI response: {response.choices[0].message.content}')
        optimized_code = response.choices[0].message.content.strip()
        score = evaluate_code_quality(optimized_code, function_name='main', optimized_code=optimized_code)

        if score > best_score:
            best_score = score;
            best_code = optimized_code

        print(f'Trial optimized code: {optimized_code} with score: {score}')

    print(f'Best optimized code: {best_code}')
    return best_code

def evaluate_code_quality(code, function_name, optimized_code):
    import subprocess
    import os
    import sys
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix='.py') as tmp:
        tmp.write(code.encode())
        tmp_path = tmp.name
        tmp_dir = os.path.dirname(tmp_path)

    sys.path.append(tmp_dir)
    module_name = os.path.splitext(os.path.basename(tmp_path))[0]
    import_statement = f'import {module_name}'

    style_result = subprocess.run(['flake8', tmp_path], capture_output=True, text=True)
    style_errors = style_result.stdout.count('\n')
    logger.info(f'Code style check result: {style_result.stdout}')

    try:
        performance_result = subprocess.run(['python', '-m', 'timeit', '-s', import_statement, f'{module_name}.{function_name}()'], capture_output=True, text=True)
        execution_time = float(performance_result.stdout.split()[0])
        logger.info(f'Code performance check result: {performance_result.stdout}')
    except IndexError:
        execution_time = float('inf')
        logger.error(f'Code performance check failed: {performance_result.stderr}')

    style_score = max(0, 100 - style_errors)
    performance_score = max(0, 100 - execution_time)
    total_score = (style_score + performance_score)

    os.unlink(tmp_path)
    logger.info(f'Code quality check result: {style_result.stdout}')
    logger.info(f'Evaluated code quality: {total_score}')
    return total_score

def again_optimize_code_with_ai(file_path):
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error('OPENAI_API_KEY 環境変数が設定されていません。')
        return None

    client = OpenAI(api_key=api_key)

    with open(file_path, 'r') as file:
        original_code = file.read()
        print(f'Optimizing code: {original_code}')

    prompt = f"""
    ### Python
    # Original Code
    {original_code}
    # Instructions
    - エラーを修正する
    """
    best_code = None
    best_score = float('-inf')

    messages = [
        {"role": "system", "content": prompt}
    ]

    for _ in range(2):
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.5
        )
        logger.info(f'OpenAI response: {response.choices[0].message.content}')
        optimized_code = response.choices[0].message.content.strip()
        score = evaluate_code_quality(optimized_code, function_name='main', optimized_code=optimized_code)

        if score > best_score:
            best_score = score;
            best_code = optimized_code

        print(f'Trial optimized code: {optimized_code} with score: {score}')

    print(f'Fix Error code: {best_code}')
    return best_code

def create_pull_request(token, repo, title, body, head, base):
    url = f"https://api.github.com/repos/{repo}/pulls"
    logger.info(f'Creating pull request: {url}')
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    logger.info(f'Headers: {headers}')
    data = {"title": title, "body": body, "head": head, "base": base}
    logger.info(f'Data: {data}')
    response = requests.post(url, headers=headers, json=data)
    logger.info(f'Response: {response.json()}')
    logger.info(f'Pull request created: {response.json()}')
    return response.json()

if __name__ == '__main__':
    app.run(port=5000, debug=True)
