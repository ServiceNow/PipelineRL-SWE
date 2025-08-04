"""
File context enrichment utilities for the unified SWE pipeline.

This module provides functionality to extract rich context from files
for use in the file selection step of the pipeline.
"""

import ast
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

import git

logger = logging.getLogger(__name__)


class FileContextEnricher:
    """Extracts rich context from source files for informed file selection."""
    
    def __init__(self):
        """Initialize the file context enricher."""
        pass
    
    def enrich_files_on_demand(
        self, 
        file_paths: List[str], 
        repo_path: Path, 
        commit_hash: str
    ) -> Dict[str, Dict]:
        """
        Extract rich context for specific files on-demand.
        
        Args:
            file_paths: List of file paths to enrich
            repo_path: Path to the local git repository
            commit_hash: Git commit hash to extract content from
            
        Returns:
            Dictionary mapping file paths to their enriched context
        """
        enriched = {}
        
        for filepath in file_paths:
            try:
                content = self._get_file_content(repo_path, commit_hash, filepath)
                if content is None:
                    logger.warning(f"Could not get content for {filepath}")
                    continue
                
                enriched[filepath] = self._extract_file_context(filepath, content)
                
            except Exception as e:
                logger.error(f"Error enriching {filepath}: {e}")
                continue
        
        return enriched
    
    def _get_file_content(self, repo_path: Path, commit_hash: str, filepath: str) -> Optional[str]:
        """
        Get file content at a specific commit.
        
        Args:
            repo_path: Path to local git repository
            commit_hash: Git commit hash
            filepath: Path to file within the repository
            
        Returns:
            File content as string, or None if error
        """
        if not os.path.exists(repo_path) or not os.path.exists(os.path.join(repo_path, '.git')):
            logger.error(f"Repository path {repo_path} does not exist or is not a git repository")
            return None

        try:
            repo = git.Repo(repo_path)
            content = repo.git.show(f'{commit_hash}:{filepath}')
            return content
        except git.exc.BadObject:
            logger.debug(f"Object '{commit_hash}:{filepath}' not found in {repo_path}")
            return None
        except git.exc.GitCommandError:
            logger.debug(f"Could not get content for {filepath} at commit {commit_hash}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting content for {filepath}: {e}")
            return None
    
    def _extract_file_context(self, filepath: str, content: str) -> Dict:
        """
        Extract rich context from file content.
        
        Args:
            filepath: Path to the file
            content: File content as string
            
        Returns:
            Dictionary with extracted context
        """
        context = {
            'content': content,
            'summary': self._create_summary(content),
            'functions': [],
            'classes': [],
            'imports': [],
            'key_lines': [],
            'file_type': self._get_file_type(filepath)
        }
        
        # Extract different contexts based on file type
        if filepath.endswith('.py'):
            context.update(self._extract_python_context(content))
        elif filepath.endswith(('.js', '.ts', '.jsx', '.tsx')):
            context.update(self._extract_javascript_context(content))
        elif filepath.endswith(('.java', '.kt')):
            context.update(self._extract_java_context(content))
        else:
            # Generic extraction for other file types
            context.update(self._extract_generic_context(content))
        
        return context
    
    def _create_summary(self, content: str, max_length: int = 500) -> str:
        """Create a summary of the file content."""
        if len(content) <= max_length:
            return content
        
        # Try to find a good break point (end of line, sentence, etc.)
        truncated = content[:max_length]
        last_newline = truncated.rfind('\n')
        if last_newline > max_length * 0.8:  # If newline is reasonably close to end
            truncated = truncated[:last_newline]
        
        return truncated + "..."
    
    def _get_file_type(self, filepath: str) -> str:
        """Determine the file type from the file path."""
        ext = Path(filepath).suffix.lower()
        
        type_mapping = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'react',
            '.tsx': 'react_typescript',
            '.java': 'java',
            '.kt': 'kotlin',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'header',
            '.hpp': 'cpp_header',
            '.cs': 'csharp',
            '.rb': 'ruby',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.html': 'html',
            '.css': 'css',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.xml': 'xml',
            '.md': 'markdown'
        }
        
        return type_mapping.get(ext, 'unknown')
    
    def _extract_python_context(self, content: str) -> Dict:
        """Extract Python-specific context using AST parsing."""
        context = {
            'functions': [],
            'classes': [],
            'imports': [],
            'key_lines': []
        }
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Extract function signature
                    args = [arg.arg for arg in node.args.args]
                    signature = f"{node.name}({', '.join(args)})"
                    context['functions'].append(signature)
                
                elif isinstance(node, ast.ClassDef):
                    context['classes'].append(node.name)
                
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        context['imports'].append(alias.name)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for alias in node.names:
                            context['imports'].append(f"{node.module}.{alias.name}")
        
        except SyntaxError:
            # Fallback to regex-based extraction if AST parsing fails
            context.update(self._extract_python_regex(content))
        
        # Extract key lines (error handling, main logic)
        context['key_lines'] = self._extract_key_lines_python(content)
        
        return context
    
    def _extract_python_regex(self, content: str) -> Dict:
        """Fallback regex-based Python extraction."""
        context = {
            'functions': [],
            'classes': [],
            'imports': []
        }
        
        # Extract function definitions
        func_pattern = r'^def\s+(\w+)\s*\([^)]*\):'
        for match in re.finditer(func_pattern, content, re.MULTILINE):
            context['functions'].append(match.group(1))
        
        # Extract class definitions
        class_pattern = r'^class\s+(\w+)(?:\([^)]*\))?:'
        for match in re.finditer(class_pattern, content, re.MULTILINE):
            context['classes'].append(match.group(1))
        
        # Extract imports
        import_pattern = r'^(?:from\s+[\w.]+\s+)?import\s+([\w.,\s]+)'
        for match in re.finditer(import_pattern, content, re.MULTILINE):
            imports = [imp.strip() for imp in match.group(1).split(',')]
            context['imports'].extend(imports)
        
        return context
    
    def _extract_key_lines_python(self, content: str) -> List[str]:
        """Extract key lines that might be relevant for debugging."""
        key_lines = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Look for error handling, assertions, main logic
            if any(keyword in stripped for keyword in [
                'raise', 'assert', 'except', 'try:', 'finally:',
                'logger.', 'print(', 'TODO', 'FIXME', 'XXX',
                'if __name__', 'main()', 'return None', 'return False'
            ]):
                key_lines.append(f"L{i+1}: {stripped}")
                
            # Limit to avoid overwhelming the context
            if len(key_lines) >= 10:
                break
        
        return key_lines
    
    def _extract_javascript_context(self, content: str) -> Dict:
        """Extract JavaScript/TypeScript context using regex."""
        context = {
            'functions': [],
            'classes': [],
            'imports': [],
            'key_lines': []
        }
        
        # Extract function definitions
        func_patterns = [
            r'function\s+(\w+)\s*\(',
            r'(\w+)\s*:\s*function\s*\(',
            r'const\s+(\w+)\s*=\s*\([^)]*\)\s*=>',
            r'(\w+)\s*\([^)]*\)\s*\{'
        ]
        
        for pattern in func_patterns:
            for match in re.finditer(pattern, content):
                context['functions'].append(match.group(1))
        
        # Extract class definitions
        class_pattern = r'class\s+(\w+)(?:\s+extends\s+\w+)?'
        for match in re.finditer(class_pattern, content):
            context['classes'].append(match.group(1))
        
        # Extract imports
        import_patterns = [
            r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]',
            r'import\s+[\'"]([^\'"]+)[\'"]',
            r'require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)'
        ]
        
        for pattern in import_patterns:
            for match in re.finditer(pattern, content):
                context['imports'].append(match.group(1))
        
        return context
    
    def _extract_java_context(self, content: str) -> Dict:
        """Extract Java/Kotlin context using regex."""
        context = {
            'functions': [],
            'classes': [],
            'imports': [],
            'key_lines': []
        }
        
        # Extract method definitions
        method_pattern = r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\([^)]*\)\s*\{'
        for match in re.finditer(method_pattern, content):
            context['functions'].append(match.group(1))
        
        # Extract class definitions
        class_pattern = r'(?:public\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w,\s]+)?'
        for match in re.finditer(class_pattern, content):
            context['classes'].append(match.group(1))
        
        # Extract imports
        import_pattern = r'import\s+([\w.]+)'
        for match in re.finditer(import_pattern, content):
            context['imports'].append(match.group(1))
        
        return context
    
    def _extract_generic_context(self, content: str) -> Dict:
        """Generic context extraction for unknown file types."""
        context = {
            'functions': [],
            'classes': [],
            'imports': [],
            'key_lines': []
        }
        
        # Try to find function-like patterns
        func_patterns = [
            r'(\w+)\s*\([^)]*\)\s*\{',  # C-style functions
            r'def\s+(\w+)',              # Python-style
            r'function\s+(\w+)',         # JavaScript-style
        ]
        
        for pattern in func_patterns:
            for match in re.finditer(pattern, content):
                context['functions'].append(match.group(1))
        
        # Look for include/import-like statements
        include_patterns = [
            r'#include\s*[<"](.*?)[>"]',  # C/C++
            r'include\s+[\'"]([^\'"]+)',  # Generic
            r'require\s+[\'"]([^\'"]+)',  # Various languages
        ]
        
        for pattern in include_patterns:
            for match in re.finditer(pattern, content):
                context['imports'].append(match.group(1))
        
        return context