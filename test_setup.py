#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys


def test_python_dependencies():
    """测试Python依赖是否安装正确"""
    # 包名和导入名的映射
    packages_to_test = [
        ("requests", "requests"),
        ("beautifulsoup4", "bs4"),  # beautifulsoup4包的导入名是bs4
        ("feedparser", "feedparser"),
        ("python-dateutil", "dateutil"),  # python-dateutil包的导入名是dateutil
        ("pytz", "pytz"),
        ("openai", "openai"),
    ]

    missing_packages = []

    for package_name, import_name in packages_to_test:
        try:
            __import__(import_name)
            print(f"✅ {package_name} - OK")
        except ImportError:
            missing_packages.append(package_name)
            print(f"❌ {package_name} - Missing")

    if missing_packages:
        print(f"\n缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install " + " ".join(missing_packages))
        return False

    return True


def test_environment_variables():
    """测试环境变量设置"""
    required_vars = ["OPENAI_API_KEY"]
    optional_vars = ["GITHUB_TOKEN", "REPO_OWNER", "REPO_NAME"]

    missing_required = []

    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var} - Set")
        else:
            missing_required.append(var)
            print(f"❌ {var} - Missing")

    for var in optional_vars:
        if os.getenv(var):
            print(f"✅ {var} - Set")
        else:
            print(f"⚠️  {var} - Not set (will be provided by GitHub Actions)")

    if missing_required:
        print(f"\n缺少必需的环境变量: {', '.join(missing_required)}")
        print('请在PowerShell中运行: $env:OPENAI_API_KEY="your-key-here"')
        return False

    return True


def test_file_structure():
    """测试文件结构是否正确"""
    required_files = [
        ".github/workflows/auto-update-r1-papers.yml",
        "scripts/scan_r1_papers.py",
        "scripts/create_pr.py",
        "requirements.txt",
    ]

    missing_files = []

    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} - Exists")
        else:
            missing_files.append(file_path)
            print(f"❌ {file_path} - Missing")

    if missing_files:
        print(f"\n缺少文件: {', '.join(missing_files)}")
        return False

    return True


def main():
    """主测试函数"""
    print("🧪 Testing R1 Papers Bot Setup...")
    print("=" * 50)

    tests = [
        ("File Structure", test_file_structure),
        ("Python Dependencies", test_python_dependencies),
        ("Environment Variables", test_environment_variables),
    ]

    all_passed = True

    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}:")
        if not test_func():
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Your R1 Papers Bot is ready to go!")
        print("\n下一步:")
        print("1. git add .")
        print("2. git commit -m '🤖 Add R1 Papers Auto-Update Bot'")
        print("3. git push origin main")
        print("4. 在GitHub仓库设置中添加OPENAI_API_KEY到Secrets")
        print("5. 启用GitHub Actions")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
