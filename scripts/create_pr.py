#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import subprocess
import requests
from datetime import datetime


class PRCreator:
    def __init__(self):
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.repo_owner = os.getenv("REPO_OWNER")
        self.repo_name = os.getenv("REPO_NAME")

        if not all([self.github_token, self.repo_owner, self.repo_name]):
            raise ValueError("Missing required environment variables")

        self.headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json",
        }

        self.base_url = (
            f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}"
        )

    def load_new_papers(self):
        """加载新发现的论文"""
        try:
            with open("new_papers.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {"added_papers": [], "count": 0}

    def setup_git(self):
        """设置Git配置"""
        subprocess.run(["git", "config", "user.name", "R1-Papers-Bot"], check=True)
        subprocess.run(["git", "config", "user.email", "action@github.com"], check=True)

    def create_branch(self, branch_name):
        """创建新分支"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--verify", branch_name],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                subprocess.run(["git", "checkout", branch_name], check=True)
                subprocess.run(["git", "pull", "origin", "main"], check=True)
            else:
                subprocess.run(["git", "checkout", "-b", branch_name], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error creating/switching branch: {e}")
            return False
        return True

    def commit_changes(self, papers):
        """提交更改"""
        commits = []

        for paper in papers:
            try:
                subprocess.run(["git", "add", "README.md"], check=True)
                commit_msg = f"Add {paper['title']}\n\nArxiv ID: {paper['arxiv_id']}\nDate: {paper['date']}"
                subprocess.run(["git", "commit", "-m", commit_msg], check=True)
                commits.append(commit_msg.split("\n")[0])
                print(f"✅ Committed: {paper['title']}")
            except subprocess.CalledProcessError as e:
                print(f"Error committing {paper['title']}: {e}")
                continue

        return commits

    def push_branch(self, branch_name):
        """推送分支到远程"""
        try:
            subprocess.run(["git", "push", "origin", branch_name], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error pushing branch: {e}")
            return False

    def create_pull_request(self, branch_name, papers, commits):
        """创建Pull Request"""
        today = datetime.now().strftime("%Y-%m-%d")

        if len(papers) == 1:
            title = f"🤖 Auto-update: Add {papers[0]['title']} ({today})"
        else:
            title = f"🤖 Auto-update: Add {len(papers)} R1 papers ({today})"

        body_lines = [
            f"## 📚 Daily R1 Papers Update - {today}",
            "",
            f"This PR automatically adds {len(papers)} new R1-related paper(s) to the Papers table.",
            "",
            "### 📋 Added Papers:",
            "",
        ]

        for i, paper in enumerate(papers, 1):
            body_lines.extend(
                [
                    f"**{i}. {paper['title']}**",
                    f"   - **ArXiv ID:** {paper['arxiv_id']}",
                    f"   - **Date:** {paper['date']}",
                    f"   - **Link:** https://arxiv.org/abs/{paper['arxiv_id']}",
                    "",
                ]
            )

        body_lines.extend(
            [
                "### 🔍 Filtering Process:",
                "- Papers are automatically discovered from arXiv CS categories",
                "- Each paper is analyzed using OpenAI GPT to ensure it's genuinely about R1 models/methods",
                "- Only papers where R1 is a core model/method name are included",
                "- Duplicate papers are automatically detected and excluded",
                "",
                "### 📝 Changes Made:",
            ]
        )

        for commit in commits:
            body_lines.append(f"- {commit}")

        body_lines.extend(
            [
                "",
                "### ✅ Ready for Review",
                "Please review the added papers and merge if they meet the quality standards.",
                "",
                "---",
                "*This PR was automatically created by the R1 Papers Bot 🤖*",
            ]
        )

        body = "\n".join(body_lines)

        pr_data = {
            "title": title,
            "head": branch_name,
            "base": "main",
            "body": body,
            "draft": False,
        }

        try:
            response = requests.post(
                f"{self.base_url}/pulls", headers=self.headers, json=pr_data
            )
            response.raise_for_status()

            pr_info = response.json()
            print(f"✅ Created PR #{pr_info['number']}: {pr_info['html_url']}")
            return pr_info

        except requests.RequestException as e:
            print(f"Error creating PR: {e}")
            if hasattr(e, "response") and e.response:
                print(f"Response: {e.response.text}")
            return None

    def check_existing_pr(self, branch_name):
        """检查是否已存在相同分支的PR"""
        try:
            response = requests.get(
                f"{self.base_url}/pulls",
                headers=self.headers,
                params={"head": f"{self.repo_owner}:{branch_name}", "state": "open"},
            )
            response.raise_for_status()

            prs = response.json()
            return len(prs) > 0

        except requests.RequestException as e:
            print(f"Error checking existing PRs: {e}")
            return False

    def run(self):
        """主运行函数"""
        print("🚀 Starting PR creation process...")

        paper_data = self.load_new_papers()

        if paper_data["count"] == 0:
            print("No new papers to create PR for")
            return

        papers = paper_data["added_papers"]
        today = datetime.now().strftime("%Y%m%d")
        branch_name = f"auto-update-r1-papers-{today}"

        if self.check_existing_pr(branch_name):
            print(f"PR for branch {branch_name} already exists")
            return

        try:
            self.setup_git()

            if not self.create_branch(branch_name):
                return

            commits = self.commit_changes(papers)

            if not commits:
                print("No commits were made")
                return

            if not self.push_branch(branch_name):
                return

            pr_info = self.create_pull_request(branch_name, papers, commits)

            if pr_info:
                print(f"🎉 Successfully created PR for {len(papers)} papers!")
            else:
                print("❌ Failed to create PR")

        except Exception as e:
            print(f"Error in PR creation process: {e}")


if __name__ == "__main__":
    creator = PRCreator()
    creator.run()
