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
        self.client = OpenAI(
            api_key=api_key,
            # base_url="https://api.openai-hub.com/v1"  # OpenAI-Hub的基础URL
            base_url="https://vip.apiyi.com/v1",  # API易的基础URL
        )

        # 高级分类prompt
        self.classification_prompt = """
You are an expert AI researcher specializing in R1-style reasoning models. Your task is to determine if a research paper is genuinely about R1-style reasoning models (following DeepSeek-R1's paradigm) or just casually mentions "R1".

CONTEXT: DeepSeek-R1 introduced a breakthrough in January 2025 for reasoning in large language models using reinforcement learning. Papers about "R1" should be related to this reasoning paradigm.

Analyze the following paper:

Title: "{title}"
Abstract: "{abstract}"

STRICT INCLUSION CRITERIA (ALL must be met):
✅ 1. NAMING PATTERN: Paper introduces/improves a model/method explicitly named with R1 pattern:
   - Valid examples: "Vision-R1", "Medical-R1", "Code-R1", "R1-Searcher", "GUI-R1"
   - Invalid: papers where R1 is just mentioned casually

✅ 2. REASONING FOCUS: Paper is about reasoning capabilities, chain-of-thought, or reinforcement learning for reasoning
   - Valid: reasoning, CoT, RLHF, reasoning chains, step-by-step thinking
   - Invalid: just using existing R1 models for applications

✅ 3. CORE CONTRIBUTION: R1 is the primary contribution/focus, not just background/comparison
   - Valid: "We propose XXX-R1", "Our R1-YYY method", "XXX-R1 outperforms"
   - Invalid: "compared with DeepSeek-R1", "inspired by R1"

STRICT EXCLUSION CRITERIA (ANY excludes the paper):
❌ 1. GENERIC R1: R1 refers to non-AI concepts:
   - "Round 1", "Region 1", "Requirement 1", "Revision 1", "Release 1"
   - "R1 approach", "R1 method" (where R1 is descriptive, not a model name)

❌ 2. CASUAL MENTION: R1 appears only in:
   - Related work section mentions
   - Comparison baselines
   - Background citations
   - "Using DeepSeek-R1 for..." applications

❌ 3. APPLICATION ONLY: Paper uses existing R1 models without novel R1 contributions
   - "We apply DeepSeek-R1 to medical diagnosis"
   - "Using R1 for text classification"

EXAMPLES FOR GUIDANCE:
✅ INCLUDE: "Vision-R1: Multimodal Reasoning with Reinforcement Learning"
✅ INCLUDE: "Medical-R1: Enhancing Clinical Reasoning via RL"
✅ INCLUDE: "R1-Code: Programming with Chain-of-Thought Reasoning"
❌ EXCLUDE: "Comparative Analysis of GPT-4 and DeepSeek-R1 on Math"
❌ EXCLUDE: "Using DeepSeek-R1 for Financial Document Analysis"
❌ EXCLUDE: "Round 1 Results of LLM Evaluation Benchmark"

DECISION PROCESS:
1. Check if paper introduces/improves a specific R1-named model/method
2. Verify it's about reasoning/RL capabilities (not just applications)
3. Confirm R1 is the core contribution (not just mentioned)
4. Ensure it's not generic R1 usage or casual mention

Respond with ONLY:
- "YES" if this paper should be included in an awesome-R1 reasoning models collection
- "NO" if it should not be included

Your response:"""

        # 信息提取prompt
        self.info_extraction_prompt = """
You are an expert at extracting structured information from research papers. Given a paper's title and abstract, extract ALL available links and information.

Paper Title: "{title}"
Abstract: "{abstract}"

Extract the following information if mentioned in the title or abstract:

1. CODE REPOSITORIES (GitHub/GitLab/etc.):
   - Main implementation code
   - Official repositories
   - Supplementary code

2. MODELS (HuggingFace/ModelScope/etc.):
   - Pre-trained models
   - Model checkpoints
   - Released model weights

3. DATASETS (HuggingFace/Kaggle/GitHub/etc.):
   - Training datasets
   - Evaluation datasets
   - Benchmark datasets

4. PROJECT PAGES:
   - Demo websites
   - Project homepages
   - Documentation sites
   - Blog posts about the project

IMPORTANT RULES:
- Only extract URLs that are EXPLICITLY mentioned in the title or abstract
- Do not infer or guess URLs
- If no URL is found for a category, leave it empty
- Prefer official/primary sources over secondary ones
- For GitHub: include full repository URLs
- For HuggingFace: include model/dataset URLs

Respond in JSON format:
{{
    "code_urls": ["url1", "url2", ...],
    "model_urls": ["url1", "url2", ...], 
    "dataset_urls": ["url1", "url2", ...],
    "project_urls": ["url1", "url2", ...]
}}

If no URLs are found for any category, use empty arrays [].
"""

    def classify_paper_with_llm(self, title: str, abstract: str) -> bool:
        """使用高级prompt进行精确分类"""
        try:
            prompt = self.classification_prompt.format(
                title=title,
                abstract=abstract[:1500],  # 限制长度但保留足够信息
            )

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert AI researcher who specializes in identifying R1-style reasoning models. You are extremely precise and only classify papers that genuinely contribute to the R1 reasoning paradigm established by DeepSeek-R1.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=10,
                temperature=0.0,  # 完全确定性
                top_p=1.0,
            )

            result = response.choices[0].message.content.strip().upper()
            is_r1_paper = result == "YES"

            # 详细日志记录
            logger.info(f"LLM Classification for: {title[:60]}...")
            logger.info(f"Result: {result}")
            if is_r1_paper:
                logger.info("✅ INCLUDED - Genuine R1 reasoning paper")
            else:
                logger.info("❌ EXCLUDED - Not a core R1 reasoning contribution")

            return is_r1_paper

        except Exception as e:
            logger.error(f"Error in LLM classification: {e}")
            logger.warning("Falling back to simple R1 pattern matching")
            return self.simple_r1_check(title, abstract)

    def extract_comprehensive_info_with_llm(
        self, title: str, abstract: str
    ) -> Dict[str, List[str]]:
        """使用LLM提取完整的论文信息"""
        try:
            prompt = self.info_extraction_prompt.format(
                title=title,
                abstract=abstract[:2000],  # 更长的摘要以获取更多信息
            )

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at extracting structured information from academic papers. You are thorough and only extract information that is explicitly mentioned.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                temperature=0.1,
            )

            result_text = response.choices[0].message.content.strip()

            # 尝试解析JSON
            try:
                github_info = json.loads(result_text)

                # 验证和清理URLs
                cleaned_info = {}
                for key in ["code_urls", "model_urls", "dataset_urls", "project_urls"]:
                    cleaned_info[key] = []
                    for url in github_info.get(key, []):
                        if isinstance(url, str) and (
                            "http://" in url or "https://" in url
                        ):
                            # 基本URL验证
                            if any(
                                domain in url
                                for domain in [
                                    "github.com",
                                    "huggingface.co",
                                    "gitlab.com",
                                    "bitbucket.org",
                                    "kaggle.com",
                                ]
                            ):
                                cleaned_info[key].append(url)

                logger.info(f"LLM extracted links: {cleaned_info}")
                return cleaned_info

            except json.JSONDecodeError:
                logger.warning(f"Failed to parse LLM response as JSON: {result_text}")
                return self.extract_links_from_text(title, abstract)

        except Exception as e:
            logger.error(f"Error extracting info with LLM: {e}")
            return self.extract_links_from_text(title, abstract)

    def simple_r1_check(self, title: str, abstract: str = "") -> bool:
        """备用的简单R1检查"""
        text = f"{title} {abstract}".lower()

        # 必须包含R1模式
        r1_patterns = [
            r"\b\w*[-_]r1\b",  # xxx-r1, xxx_r1
            r"\br1[-_]\w*\b",  # r1-xxx, r1_xxx
            r"\b\w+r1\b",  # visionr1, coder1
            r"\br1\w+\b",  # r1vision, r1code
        ]

        has_r1 = any(re.search(pattern, text) for pattern in r1_patterns)
        if not has_r1:
            return False

        # 排除明显不相关的
        exclude_patterns = [
            r"\bround\s*1\b",
            r"\bregion\s*1\b",
            r"\brevision\s*1\b",
            r"\brequirement\s*1\b",
            r"\brelease\s*1\b",
        ]

        is_excluded = any(re.search(pattern, text) for pattern in exclude_patterns)
        if is_excluded:
            return False

        # 必须有AI/ML上下文
        ml_keywords = [
            "model",
            "neural",
            "learning",
            "reasoning",
            "language",
            "vision",
            "multimodal",
        ]
        has_ml_context = any(keyword in text for keyword in ml_keywords)

        return has_ml_context

    def extract_links_from_text(
        self, title: str, abstract: str
    ) -> Dict[str, List[str]]:
        """从论文文本中提取GitHub等链接（备用方案）"""
        text_to_search = f"{title} {abstract}"

        # 提取各类链接
        github_matches = re.findall(r"https://github\.com/[^\s\)\]]+", text_to_search)
        huggingface_matches = re.findall(
            r"https://huggingface\.co/[^\s\)\]]+", text_to_search
        )
        other_matches = re.findall(r"https://[^\s\)\]]+", text_to_search)

        links = {
            "code_urls": [],
            "model_urls": [],
            "dataset_urls": [],
            "project_urls": [],
        }

        # 分类GitHub链接
        for url in github_matches:
            links["code_urls"].append(url)

        # 分类HuggingFace链接
        for url in huggingface_matches:
            if "/models/" in url or "/model/" in url:
                links["model_urls"].append(url)
            elif "/datasets/" in url or "/dataset/" in url:
                links["dataset_urls"].append(url)
            else:
                links["model_urls"].append(url)  # 默认归类为模型

        # 分类其他链接
        for url in other_matches:
            if url not in github_matches and url not in huggingface_matches:
                if any(domain in url for domain in ["demo", "project", "page", "site"]):
                    links["project_urls"].append(url)

        return links

    def search_arxiv_papers(self, days_back: int = 3) -> List[Dict]:
        """搜索arXiv中最近的R1相关论文（过去3天）"""
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

                        # 处理日期 - 使用第一个版本的上传时间
                        published_date = date_parser.parse(entry.published)

                        # 移除版本号获取基础ID
                        arxiv_id_base = (
                            arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                        )

                        authors = []
                        if hasattr(entry, "authors"):
                            authors = [author.name for author in entry.authors]
                        elif hasattr(entry, "author"):
                            authors = [entry.author]

                        # 使用LLM提取完整信息
                        links = self.extract_comprehensive_info_with_llm(
                            title, abstract
                        )

                        paper_info = {
                            "title": title,
                            "arxiv_id": arxiv_id_base,  # 使用基础ID（无版本号）
                            "authors": authors,
                            "published": published_date.strftime("%Y.%m.%d"),
                            "summary": abstract,
                            "arxiv_url": f"https://arxiv.org/abs/{arxiv_id_base}",
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

        # 去重 - 基于arxiv_id（无版本号）
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

            # 更精确地提取已存在的arXiv ID
            arxiv_links = re.findall(
                r"https://arxiv\.org/abs/([0-9]+\.[0-9]+)", content
            )
            # 同时提取版本号形式的ID（如2401.12345v1）
            versioned_ids = []
            for arxiv_id in arxiv_links:
                base_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                versioned_ids.append(base_id)

            existing_arxiv_ids.update(arxiv_links)
            existing_arxiv_ids.update(versioned_ids)

            logger.info(f"Found {len(existing_arxiv_ids)} existing papers in README.md")
            return existing_arxiv_ids
        except FileNotFoundError:
            logger.info("README.md not found")
            return set()

    def format_table_row(self, paper: Dict) -> str:
        """格式化表格行，按照awesome-R1仓库的风格"""
        links = paper["links"]

        # 构建Paper列 - 包含标题和arXiv链接
        paper_cell = f"[{paper['title']}]({paper['arxiv_url']})"

        # 构建各个链接列，支持多个链接
        def format_links(urls: List[str], default_name: str) -> str:
            if not urls:
                return "-"

            formatted_links = []
            for i, url in enumerate(urls):
                # 从URL中提取有意义的名称
                if "github.com" in url:
                    parts = url.split("/")
                    if len(parts) >= 5:
                        repo_name = f"{parts[-2]}/{parts[-1]}".replace(".git", "")
                        link_text = repo_name if len(urls) == 1 else f"{repo_name}"
                    else:
                        link_text = (
                            f"{default_name}{i + 1}" if len(urls) > 1 else default_name
                        )
                elif "huggingface.co" in url:
                    parts = url.split("/")
                    if len(parts) >= 5:
                        model_name = "/".join(parts[-2:])
                        link_text = model_name if len(urls) == 1 else f"{model_name}"
                    else:
                        link_text = (
                            f"{default_name}{i + 1}" if len(urls) > 1 else default_name
                        )
                else:
                    # 其他类型的链接
                    link_text = (
                        f"{default_name}{i + 1}" if len(urls) > 1 else default_name
                    )

                formatted_links.append(f"[{link_text}]({url})")

            # 如果有多个链接，用<br/>分隔；否则直接返回
            return "<br/>".join(formatted_links)

        code_cell = format_links(links["code_urls"], "Code")
        models_cell = format_links(links["model_urls"], "Models")
        dataset_cell = format_links(links["dataset_urls"], "Dataset")
        project_cell = format_links(links["project_urls"], "Project")

        # 生成表格行
        row = f"| {paper_cell} | {code_cell} | {models_cell} | {dataset_cell} | {project_cell} | {paper['published']} |"

        return row

    def update_readme(
        self, new_papers: List[Dict], readme_path: str = "README.md"
    ) -> List[Dict]:
        """精确更新README.md - 只修改Papers表格的特定位置"""
        if not new_papers:
            logger.info("No new papers to add")
            return []

        try:
            with open(readme_path, "r", encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            logger.error("README.md not found")
            return []

        # 精确匹配Papers表格的位置
        # 查找表格开始和结束位置
        table_start_pattern = (
            r"(## Papers\s*\n\s*\| Paper.*?\| Date.*?\|\n\| -+.*?\|\n)"
        )
        table_end_pattern = r"(\n_Sort by.*?$|\n## [^P].*?$|\Z)"

        start_match = re.search(table_start_pattern, content, re.MULTILINE)
        if not start_match:
            logger.error("Could not find Papers table header")
            return []

        # 找到表格数据开始位置
        table_start_pos = start_match.end()

        # 找到表格结束位置
        remaining_content = content[table_start_pos:]
        end_match = re.search(table_end_pattern, remaining_content, re.MULTILINE)

        if end_match:
            table_end_pos = table_start_pos + end_match.start()
            existing_table_data = content[table_start_pos:table_end_pos]
            after_table = content[table_end_pos:]
        else:
            existing_table_data = remaining_content
            after_table = ""

        # 生成新的表格行
        new_rows = []
        added_papers = []

        for paper in new_papers:
            new_row = self.format_table_row(paper)
            new_rows.append(new_row)
            added_papers.append(paper)
            logger.info(f"Adding paper: {paper['title']}")

        # 构建新的表格内容
        table_header = start_match.group(1)
        new_table_data = "\n".join(new_rows)

        if existing_table_data.strip():
            complete_table = table_header + new_table_data + "\n" + existing_table_data
        else:
            complete_table = table_header + new_table_data + "\n"

        # 重新组装完整内容
        before_table = content[: start_match.start()]
        new_content = before_table + complete_table + after_table

        # 写入文件
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        logger.info(
            f"Successfully updated README.md with {len(added_papers)} new papers"
        )
        return added_papers

    def run(self):
        """主运行函数"""
        logger.info("🤖 Starting R1 Paper Scanner...")

        existing_papers = self.load_existing_papers()
        all_papers = self.search_arxiv_papers(days_back=3)  # 搜索过去3天

        new_papers = [
            paper for paper in all_papers if paper["arxiv_id"] not in existing_papers
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
