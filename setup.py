#!/usr/bin/env python3
"""
Setup for ASI V2.5 Hugging Face Package
🚀 2.44x speedup demonstrated on Longformer
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="asi-v25-longformer",
    version="2.5.0",
    author="ASI Research Team", 
    author_email="contact@asi-research.com",
    description="Ultra-Professional Linear Attention with 2.44x speedup",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/asi-research/asi-v25-longformer-core",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research", 
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "transformers>=4.21.0",
        "numpy>=1.21.0",
        "datasets>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
        ],
        "demo": [
            "gradio>=3.0",
            "matplotlib>=3.5.0",
            "plotly>=5.0",
        ],
        "enterprise": [
            "flash-attn>=2.0",
            "triton>=2.0",
        ],
    },
    keywords=[
        "attention-mechanism", 
        "linear-attention",
        "longformer", 
        "performance-optimization",
        "transformer",
        "pytorch",
        "natural-language-processing",
        "deep-learning",
    ],
    project_urls={
        "Bug Tracker": "https://github.com/asi-research/asi-v25-longformer-core/issues",
        "Documentation": "https://huggingface.co/asi-research/asi-v25-longformer-core",
        "Source Code": "https://github.com/asi-research/asi-v25-longformer-core",
        "Demo Space": "https://huggingface.co/spaces/asi-research/asi-v25-live-demo",
        "Enterprise": "https://asi-research.com/enterprise",
    },
)
