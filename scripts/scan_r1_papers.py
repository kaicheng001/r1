#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import requests
import feedparser
from datetime import datetime, timedelta
from dateutil import parser as date_parser
import pytz
import time
from openai import OpenAI
from typing import List, Dict, Optional
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class R1PaperScanner:
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
        self.cs_categories = [
            "cs.AI",
            "cs.CL",
            "cs.LG",
            "cs.CV",
            "cs.CY",
            "cs.DC",
            "cs.DS",
            "cs.GT",
            "cs.HC",
            "cs.IR",
            "cs.IT",
            "cs.LO",
            "cs.MA",
            "cs.MM",
            "cs.NI",
            "cs.NE",
            "cs.PL",
            "cs.RO",
            "cs.SE",
            "cs.SI",
            "cs.SY",
        ]

        # 初始化OpenAI客户端
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self.client = OpenAI(api_key=api_key)

        self.classification_prompt = """
You are an expert in machine learning and AI research. Your task is to determine if a research paper is about R1-style reasoning models, following the pattern established by DeepSeek-R1.

Analyze the following paper title and abstract to determine if:

✅ INCLUDE if:
1. The paper introduces a new R1-style model/method (e.g., "XXX-R1", "R1-XXX")
2. R1 is the primary focus and represents a reasoning approach similar to DeepSeek-R1
3. The paper builds upon or extends R1-style reasoning methodology
4. R1 appears as a specific model/method name in the title

❌ EXCLUDE if:
1. R1 is mentioned only casually or in comparisons
2. R1 refers to generic concepts like "round 1", "region 1", "requirement 1"
3. The paper is primarily about other methods that happen to mention R1
4. R1 appears only in references or background sections

Title: "{title}"
Abstract: "{abstract}"

Respond with only "YES" if this paper should be included in the awesome-R1 collection, or "NO" if it should not be included.
"""

    def classify_paper_with_llm(self, title: str, abstract: str) -> bool:
        """使用OpenAI API判断论文是否应该被包含"""
        try:
            prompt = self.classification_prompt.format(
                title=title, abstract=abstract[:1200]
            )

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert AI researcher specializing in R1-style reasoning models.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=10,
                temperature=0.1,
            )

            result = response.choices[0].message.content.strip().upper()
            is_r1_paper = result == "YES"

            logger.info(f"LLM Classification: {title[:50]}... -> {result}")
            return is_r1_paper

        except Exception as e:
            logger.error(f"Error in LLM classification: {e}")
            return self.simple_r1_check(title)

    def simple_r1_check(self, title: str) -> bool:
        """简单的R1模式检查作为备用方案"""
        r1_patterns = [
            r"\b[A-Za-z0-9]*[-_]?R1\b",
            r"\b[A-Za-z0-9]*[-_]?r1\b",
            r"\bR1[-_][A-Za-z0-9]*\b",
            r"\br1[-_][A-Za-z0-9]*\b",
        ]

        for pattern in r1_patterns:
            if re.search(pattern, title, re.IGNORECASE):
                return True
        return False

    def extract_links_from_text(self, title: str, abstract: str) -> Dict[str, str]:
        """从论文文本中提取GitHub等链接"""
        text_to_search = f"{title} {abstract}"

        # 提取GitHub链接
        github_matches = re.findall(r"https://github\.com/[^\s\)\]]+", text_to_search)
        huggingface_matches = re.findall(
            r"https://huggingface\.co/[^\s\)\]]+", text_to_search
        )

        links = {"code": "", "models": "", "dataset": "", "project": ""}

        # 简单分类链接
        for url in github_matches:
            if not links["code"]:
                links["code"] = url

        for url in huggingface_matches:
            if "models" in url or "checkpoint" in url:
                if not links["models"]:
                    links["models"] = url
            elif "datasets" in url:
                if not links["dataset"]:
                    links["dataset"] = url

        return links

    def search_arxiv_papers(self, days_back: int = 1) -> List[Dict]:
        """搜索arXiv中最近的R1相关论文"""
        papers = []

        end_date = datetime.now(pytz.UTC)
        start_date = end_date - timedelta(days=days_back)

        logger.info(
            f"Searching papers from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )

        for category in self.cs_categories:
            logger.info(f"Searching in category: {category}")

            query = f"cat:{category} AND submittedDate:[{start_date.strftime('%Y%m%d')}0000 TO {end_date.strftime('%Y%m%d')}2359]"
            params = {
                "search_query": query,
                "start": 0,
                "max_results": 50,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }

            try:
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()

                feed = feedparser.parse(response.content)

                for entry in feed.entries:
                    title = entry.title.strip()
                    abstract = (
                        entry.summary.strip() if hasattr(entry, "summary") else ""
                    )

                    if self.classify_paper_with_llm(title, abstract):
                        arxiv_id = entry.id.split("/")[-1]
                        # published_date = date_parser.parse(entry.published)
                        # 处理日期 - 使用第一个版本的上传时间
                        published_date = date_parser.parse(entry.published)
                        # 如果有更早版本，获取最早版本的时间
                        arxiv_id_base = arxiv_id.split('v')[0]  # 移除版本号
                        # 使用arxiv API获取第一版本时间（简化处理，直接使用published时间）

                        authors = []
                        if hasattr(entry, "authors"):
                            authors = [author.name for author in entry.authors]
                        elif hasattr(entry, "author"):
                            authors = [entry.author]

                        links = self.extract_links_from_text(title, abstract)

                        paper_info = {
                            "title": title,
                            "arxiv_id": arxiv_id,
                            "authors": authors,
                            "published": published_date.strftime("%Y.%m.%d"),
                            "summary": abstract,
                            "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
                            "links": links,
                        }

                        papers.append(paper_info)
                        logger.info(f"✅ Found R1 paper: {title}")

                    time.sleep(0.8)

                time.sleep(2)

            except Exception as e:
                logger.error(f"Error searching category {category}: {e}")
                continue

        papers = sorted(papers, key=lambda x: x["published"], reverse=True)

        # 去重
        seen_ids = set()
        unique_papers = []
        for paper in papers:
            if paper["arxiv_id"] not in seen_ids:
                seen_ids.add(paper["arxiv_id"])
                unique_papers.append(paper)

        logger.info(f"Found {len(unique_papers)} unique R1 papers")
        return unique_papers

    def load_existing_papers(self, readme_path: str = "README.md") -> set:
        """从README.md中加载已存在的论文"""
        existing_arxiv_ids = set()

        try:
            with open(readme_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 提取arxiv链接中的ID
            arxiv_links = re.findall(
                r"https://arxiv\.org/abs/([0-9]+\.[0-9]+)", content
            )
            existing_arxiv_ids.update(arxiv_links)

        except FileNotFoundError:
            logger.info("README.md not found")

        logger.info(f"Found {len(existing_arxiv_ids)} existing papers in README.md")
        return existing_arxiv_ids

    def format_table_row(self, paper: Dict) -> str:
        """格式化表格行，按照awesome-R1仓库的风格"""
        links = paper["links"]

        # 构建Paper列 - 包含标题和arXiv链接
        paper_cell = f"[{paper['title']}]({paper['arxiv_url']})"

        # 构建各个链接列
        code_cell = (
            f"[{links['code'].split('/')[-1]}]({links['code']})"
            if links["code"]
            else "-"
        )
        models_cell = (
            f"[{links['models'].split('/')[-1]}]({links['models']})"
            if links["models"]
            else "-"
        )
        dataset_cell = (
            f"[{links['dataset'].split('/')[-1]}]({links['dataset']})"
            if links["dataset"]
            else "-"
        )
        project_cell = (
            f"[{links['project'].split('/')[-1]}]({links['project']})"
            if links["project"]
            else "-"
        )

        # 生成表格行
        row = f"| {paper_cell} | {code_cell} | {models_cell} | {dataset_cell} | {project_cell} | {paper['published']} |"

        return row

    def update_readme(
        self, new_papers: List[Dict], readme_path: str = "README.md"
    ) -> List[Dict]:
        """更新README.md文件 - 在Papers表格第一行添加新论文"""
        if not new_papers:
            logger.info("No new papers to add")
            return []

        try:
            with open(readme_path, "r", encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            logger.error("README.md not found")
            return []

        # 查找Papers表格位置
        table_pattern = r"(## Papers\s*\n\s*\| Paper.*?\| Date.*?\|\n\| -+.*?\|\n)(.*?)(?=\n## |\n_Sort by|\Z)"
        match = re.search(table_pattern, content, re.DOTALL)

        if not match:
            logger.error("Could not find Papers table in README.md")
            return []

        table_header = match.group(1)
        existing_rows = match.group(2).strip()

        # 为每篇新论文生成表格行
        new_rows = []
        added_papers = []

        for paper in new_papers:
            new_row = self.format_table_row(paper)
            new_rows.append(new_row)
            added_papers.append(paper)

        # 在表格开头插入新行
        if existing_rows:
            updated_table = table_header + "\n".join(new_rows) + "\n" + existing_rows
        else:
            updated_table = table_header + "\n".join(new_rows)

        # 替换原有表格
        new_content = content[: match.start()] + updated_table + content[match.end() :]

        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        logger.info(f"Updated README.md with {len(added_papers)} new papers")
        return added_papers

    def run(self):
        """主运行函数"""
        logger.info("🤖 Starting R1 Paper Scanner...")
        
        existing_papers = self.load_existing_papers()
        all_papers = self.search_arxiv_papers(days_back=3)  # 改为3天
        
        new_papers = [
            paper for paper in all_papers 
            if paper['arxiv_id'] not in existing_papers
        ]

        if not new_papers:
            logger.info("No new R1 papers found")
            return

        logger.info(f"Found {len(new_papers)} new R1 papers")

        added_papers = self.update_readme(new_papers)

        # 保存结果供PR脚本使用
        result = {
            "added_papers": [
                {
                    "title": paper["title"],
                    "arxiv_id": paper["arxiv_id"],
                    "date": paper["published"],
                }
                for paper in added_papers
            ],
            "count": len(added_papers),
            "scan_date": datetime.now().isoformat(),
        }

        with open("new_papers.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info("✅ R1 Paper Scanner completed")


if __name__ == "__main__":
    scanner = R1PaperScanner()
    scanner.run()
