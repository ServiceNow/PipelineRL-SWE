"""
SWE-Gym dataset preprocessing utility.

This module provides functionality to preprocess the SWE-Gym dataset by:
1. Cloning or updating the required GitHub repositories
2. Extracting file contents at the specified commits
3. Enriching the dataset with these file contents
4. Computing file statistics for ALL files in repo for BM25 retrieval
5. Filtering based on token thresholds (min and max)
6. Saving the processed dataset for use with PipelineRL
"""

import os
import re
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter

import git
import hydra
from datasets import load_dataset, load_from_disk, Dataset
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class RepoManager:
    """Manages the cloning and updating of repositories for SWE-Gym dataset."""
    
    def __init__(self, repos_base_dir: str):
        """
        Initialize the repository manager.
        
        Args:
            repos_base_dir: Directory where repositories will be stored
        """
        self.repos_base_dir = Path(repos_base_dir)
        os.makedirs(self.repos_base_dir, exist_ok=True)
        logger.info(f"Repository directory '{self.repos_base_dir}' ensured.")
        
    def clone_or_update_repo(self, repo_name: str) -> Path:
        """
        Clones or updates a Git repository.
        
        Args:
            repo_name: Name of the repository in format "owner/repo"
            
        Returns:
            Path to the local repository
        """
        repo_url = f"https://github.com/{repo_name}.git"
        # Sanitize repo_name for local path (replace / with _)
        local_path = os.path.join(self.repos_base_dir, repo_name.replace("/", "_"))

        try:
            if os.path.exists(local_path) and os.path.exists(os.path.join(local_path, '.git')):
                # If the directory exists and looks like a repo, update it
                logger.debug(f"Updating repository: {repo_name} at {local_path}")
                repo = git.Repo(local_path)
                origin = repo.remotes.origin
                # Fetch all updates from origin
                origin.fetch()
            else:
                # If directory doesn't exist or isn't a valid repo, clone fresh
                # Clean up potentially broken directory first
                if os.path.exists(local_path):
                    logger.warning(f"Directory {local_path} exists but is not a valid git repo. Removing and cloning fresh.")
                    try:
                        shutil.rmtree(local_path)
                    except Exception as cleanup_e:
                        logger.error(f"Failed to remove directory {local_path}: {cleanup_e}")

                logger.debug(f"Cloning repository: {repo_name} to {local_path}")
                # A full clone is generally safer to ensure base_commit exists
                git.Repo.clone_from(repo_url, local_path)
                logger.debug(f"Cloned repository: {repo_name}")

            return Path(local_path)
        except git.exc.GitCommandError as e:
            logger.error(f"Git command failed for {repo_name} ({local_path}): {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to process {repo_name} ({local_path}): {e}")
            raise
            
    def clone_or_update_repos(self, repo_names: list[str]) -> dict[str, Path]:
        """
        Clone or update multiple repositories.
        
        Args:
            repo_names: List of repository names
            
        Returns:
            Dictionary mapping repository names to local paths
        """
        failed_repos = []
        processed_repos = {}
        
        for repo_name in tqdm(repo_names, desc="Cloning/Updating Repos"):
            try:
                local_path = self.clone_or_update_repo(repo_name)
                # Verify if the repo is now a valid git directory after attempting clone/update
                if os.path.exists(os.path.join(local_path, '.git')):
                    processed_repos[repo_name] = local_path
                else:
                    # If it's not a valid repo after clone/update attempt
                    failed_repos.append(repo_name)
                    logger.warning(f"Repository {repo_name} at {local_path} is not a valid git repo after processing.")
            except Exception as e:
                logger.error(f"An unexpected error occurred processing repo {repo_name}: {e}")
                failed_repos.append(repo_name)
                
        if failed_repos:
            # Filter out duplicates in case of multiple failures/warnings for the same repo
            failed_repos = list(set(failed_repos))
            logger.warning(f"Failed to clone/update or validate {len(failed_repos)} repositories: {failed_repos}. Examples from these repos may be skipped during mapping.")
            
        logger.info("Repository cloning/updating process finished.")
        return processed_repos


class SwePreprocessor:
    """Processes the SWE-Gym dataset to extract file contents at specific commits."""
    
    def __init__(self, cfg: DictConfig):
        """
        Initialize the dataset processor.
        
        Args:
            cfg: Hydra configuration containing all necessary parameters
        """
        self.cfg = cfg
        self.repos_base_dir = Path(cfg.swe.repos_base_dir)
        self.dataset_path = Path(cfg.swe.dataset_path)
        self.min_token_threshold = cfg.swe.min_token_threshold
        self.max_token_threshold = cfg.swe.max_token_threshold
        self.num_map_processes = cfg.swe.num_map_processes
        self.tokenizer_model = cfg.swe.tokenizer_model
        
        # Initialize tokenizer
        self.tokenizer = None
        self._init_tokenizer()
        
        # Initialize repo manager
        self.repo_manager = RepoManager(self.repos_base_dir)
        
        # Cache for file stats to avoid recomputing for same (repo, commit) pairs
        self.file_stats_cache = {}
        
        # Define file extensions to include (source code files)
        self.source_extensions = {
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp', '.cs', '.php',
            '.rb', '.go', '.rs', '.kt', '.scala', '.swift', '.m', '.mm', '.sh', '.bash',
            '.zsh', '.fish', '.pl', '.r', '.R', '.sql', '.html', '.css', '.scss', '.sass',
            '.less', '.vue', '.jsx', '.tsx', '.json', '.yaml', '.yml', '.xml', '.toml',
            '.ini', '.cfg', '.conf', '.properties', '.gradle', '.cmake', '.make',
            '.dockerfile', '.md', '.rst', '.txt', '.lock', '.requirements'
        }
        
        # Define directories to skip
        self.skip_directories = {
            'test', 'tests', '__pycache__', '.git', '.svn', '.hg', 'node_modules',
            '.pytest_cache', '.tox', 'venv', 'env', '.env', 'build', 'dist',
            '.idea', '.vscode', 'target', 'out', 'bin', 'obj', '.gradle',
            'coverage', '.coverage', '.nyc_output', 'htmlcov'
        }
        
    def _init_tokenizer(self):
        """Initialize the tokenizer for token counting."""
        logger.info(f"Attempting to load tokenizer: {self.tokenizer_model}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model)
            logger.info("Tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load tokenizer {self.tokenizer_model}: {e}")
            logger.warning("Tokenization statistics will NOT be calculated. Please ensure transformers library is installed and you have access to the model if it's gated.")
            self.tokenizer = None

    def _tokenize_content(self, content: str) -> Counter:
        """
        Tokenize file content for BM25 retrieval.
        
        Args:
            content: File content as string
            
        Returns:
            Counter object with token frequencies
        """
        if not content:
            return Counter()
            
        # Split on non-alphanumeric characters (keeping underscores)
        tokens = re.findall(r'[a-zA-Z0-9_]+', content.lower())
        
        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            # Skip very short or very long tokens
            if len(token) < 2 or len(token) > 50:
                continue
            # Skip tokens that are all digits (unless they're meaningful like version numbers)
            if token.isdigit() and len(token) > 4:
                continue
            filtered_tokens.append(token)
            
        return Counter(filtered_tokens)

    def _is_source_file(self, filepath: str) -> bool:
        """
        Check if a file should be included based on extension and path.
        
        Args:
            filepath: Path to the file
            
        Returns:
            True if file should be included
        """
        path = Path(filepath)
        
        # Check if any parent directory should be skipped
        for part in path.parts:
            if part.lower() in self.skip_directories:
                logger.debug(f"DEBUG: Skipping {filepath} due to directory {part}")
                return False
                
        # Check extension
        if path.suffix.lower() in self.source_extensions:
            logger.debug(f"DEBUG: Including {filepath} due to extension {path.suffix}")
            return True
            
        # Include files without extensions that might be source files
        if not path.suffix and path.name.lower() in {
            'makefile', 'dockerfile', 'readme', 'license', 'changelog',
            'requirements', 'pipfile', 'gemfile', 'rakefile'
        }:
            logger.debug(f"DEBUG: Including {filepath} due to special name {path.name}")
            return True
        
        logger.debug(f"DEBUG: Excluding {filepath} - no matching criteria")
        return False

    def _is_text_file(self, content_bytes: bytes) -> bool:
        """
        Check if file content is text (not binary).
        
        Args:
            content_bytes: File content as bytes
            
        Returns:
            True if file appears to be text
        """
        try:
            content_bytes.decode('utf-8')
            return True
        except UnicodeDecodeError:
            return False

    def _get_all_file_stats(self, local_repo_path: Path, commit_hash: str) -> Dict[str, Dict]:
        """
        Get statistics for all source files in the repository at a specific commit.
        
        Args:
            local_repo_path: Path to local git repository
            commit_hash: Git commit hash
            
        Returns:
            Dictionary mapping file paths to their statistics
        """
        cache_key = (str(local_repo_path), commit_hash)
        if cache_key in self.file_stats_cache:
            return self.file_stats_cache[cache_key]
            
        file_stats = {}
        
        if not os.path.exists(local_repo_path) or not os.path.exists(os.path.join(local_repo_path, '.git')):
            self.file_stats_cache[cache_key] = file_stats
            return file_stats

        try:
            repo = git.Repo(local_repo_path)
            
            # Get all files at the specific commit using ls-tree
            try:
                # Get all files recursively at the commit
                files_output = repo.git.execute(['git', 'ls-tree', '-r', '--name-only', commit_hash])
                all_files = files_output.strip().split('\n') if files_output.strip() else []
            except git.exc.GitCommandError as e:
                logger.debug(f"Could not list files at commit {commit_hash} in {local_repo_path}: {e}")
                self.file_stats_cache[cache_key] = file_stats
                return file_stats
            
            # Process each file
            processed_count = 0
            source_files_found = []  # DEBUG: Track accepted files
            python_files_found = []  # DEBUG: Track Python files specifically
            for filepath in all_files:
                if not filepath:  # Skip empty lines
                    continue
                    
                # Check if this is a source file we want to include
                if not self._is_source_file(filepath):
                    continue
                
                source_files_found.append(filepath)  # DEBUG: Track this
                if filepath.endswith('.py'):
                    python_files_found.append(filepath)
                
                try:
                    # Get file content at the specific commit
                    content = repo.git.show(f'{commit_hash}:{filepath}')
                    
                    # Check if it's a text file by trying to encode/decode
                    try:
                        content_bytes = content.encode('utf-8')
                        if not self._is_text_file(content_bytes):
                            logger.debug(f"DEBUG: Skipping binary file {filepath}")
                            continue
                    except UnicodeDecodeError:
                        logger.debug(f"DEBUG: Skipping non-UTF8 file {filepath}")
                        continue
                    
                    # Calculate statistics
                    term_counts = self._tokenize_content(content)
                    
                    file_stats[filepath] = {
                        'path': filepath,
                        'length': len(content),
                        'term_counts': dict(term_counts)  # Convert Counter to dict for JSON serialization
                    }
                    
                    processed_count += 1
                    logger.debug(f"DEBUG: Successfully processed {filepath}, length={len(content)}")
                    
                    # Limit number of files per repo to avoid excessive processing
                    if processed_count >= 1000:  # Configurable limit
                        logger.debug(f"Reached file limit (1000) for repo {local_repo_path} at commit {commit_hash}")
                        break
                        
                except git.exc.GitCommandError as e:
                    # File might not exist at this commit or other git error
                    logger.debug(f"DEBUG: Git error for {filepath}: {e}")
                    continue
                except Exception as e:
                    logger.debug(f"Error processing file {filepath} at commit {commit_hash}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error getting file stats for {local_repo_path} at commit {commit_hash}: {e}")
            
        # Cache the results
        self.file_stats_cache[cache_key] = file_stats
        logger.info(f"DEBUG: Found {len(source_files_found)} source files, {len(python_files_found)} Python files, processed {len(file_stats)} files for {local_repo_path} at commit {commit_hash}")
        if python_files_found:
            logger.info(f"DEBUG: First 5 Python files: {python_files_found[:5]}")
        elif source_files_found:
            logger.info(f"DEBUG: First 5 source files: {source_files_found[:5]}")
        
        return file_stats

    def _parse_patch(self, patch_string: str) -> List[str]:
        """
        Extracts the 'a/' file paths from a standard git patch string.
        These correspond to the file paths BEFORE the patch is applied.
        
        Args:
            patch_string: Git patch string
            
        Returns:
            List of file paths
        """
        if not patch_string:
            return []
        # Regex to find lines starting with '--- a/' and capture the path following 'a/'
        gold_filepaths = re.findall(r'^--- a/(.+)$', patch_string, re.MULTILINE)
        return gold_filepaths

    def _get_file_content(self, local_repo_path: Path, commit_hash: str, filepath: str) -> Optional[str]:
        """
        Gets the content of a specific file at a specific commit hash from a local Git repository.
        
        Args:
            local_repo_path: Path to local git repository
            commit_hash: Git commit hash
            filepath: Path to file within the repository
            
        Returns:
            File content as string, or None if file not found or error
        """
        # Basic check for valid repo path before attempting git commands
        if not os.path.exists(local_repo_path) or not os.path.exists(os.path.join(local_repo_path, '.git')):
            return None

        try:
            repo = git.Repo(local_repo_path)
            # Use git show commit:filepath to get the content
            content = repo.git.show(f'{commit_hash}:{filepath}')
            return content
        except git.exc.BadObject:
            # Commit or object (file) not found at that commit
            logger.debug(f"Object '{commit_hash}:{filepath}' not found in {local_repo_path}. Skipping.")
            return None
        except git.exc.GitCommandError:
            # This can happen if the file doesn't exist at that commit or other git errors
            logger.debug(f"Could not get content for {filepath} at commit {commit_hash} in {local_repo_path}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred getting content for {filepath} at commit {commit_hash} in {local_repo_path}: {e}")
            return None

    def _process_example(self, example: Dict, repo_paths: Dict[str, Path]) -> Dict:
        """
        Processes a single dataset example to extract gold file contents and all file stats at the base_commit.
        
        Args:
            example: Dataset example
            repo_paths: Dictionary mapping repository names to local paths
            
        Returns:
            Processed example with file contents and file stats added
        """
        repo_name = example.get('repo')
        base_commit = example.get('base_commit')
        patch = example.get('patch')

        # Ensure we have the necessary data fields
        if not repo_name or not base_commit or not patch:
            example['gold_file_contents'] = json.dumps({})
            example['all_file_stats'] = json.dumps({})
            return example

        # Get local repo path
        local_repo_path = repo_paths.get(repo_name)
        if not local_repo_path:
            logger.debug(f"Repository {repo_name} not found in processed repos.")
            example['gold_file_contents'] = json.dumps({})
            example['all_file_stats'] = json.dumps({})
            return example

        # Skip processing if the repo directory doesn't exist or isn't a valid git repo
        if not os.path.exists(local_repo_path) or not os.path.exists(os.path.join(local_repo_path, '.git')):
            logger.debug(f"Local repository directory missing or invalid for {repo_name}. Skipping file content extraction for this example.")
            example['gold_file_contents'] = json.dumps({})
            example['all_file_stats'] = json.dumps({})
            return example

        # Extract file paths from patch (gold files)
        gold_filepaths = self._parse_patch(patch)
        file_contents_dict = {}

        # Get content for each gold file
        for filepath in gold_filepaths:
            content = self._get_file_content(local_repo_path, base_commit, filepath)
            if content is not None:
                file_contents_dict[filepath] = content

        # Store gold file contents as JSON string
        example['gold_file_contents'] = json.dumps(file_contents_dict)
        
        # Get statistics for all files in the repository
        all_file_stats = self._get_all_file_stats(local_repo_path, base_commit)
        example['all_file_stats'] = json.dumps(all_file_stats)
        
        return example

    def _filter_by_token_count(self, example: Dict) -> bool:
        """
        Filter function to determine if an example should be kept based on token count.
        
        Args:
            example: Dataset example with gold_file_contents
            
        Returns:
            True if example should be kept, False if it should be filtered out
        """
        # If we don't have a tokenizer, we can't filter by token count
        if not self.tokenizer:
            return True
            
        try:
            file_contents_dict = json.loads(example.get('gold_file_contents', '{}'))
            
            # If no file contents, skip the example
            if not file_contents_dict:
                return False
                
            all_gold_text = " ".join(file_contents_dict.values())
            all_gold_text_w_pr_s = f"{all_gold_text} {example.get('problem_statement')}"
            tokens = self.tokenizer.encode(all_gold_text_w_pr_s, add_special_tokens=False)
            tokens_without_ps = self.tokenizer.encode(all_gold_text, add_special_tokens=False)
            token_count = len(tokens)
            
            # Keep example if token count is between min and max thresholds
            return (token_count >= self.min_token_threshold and 
                   (self.max_token_threshold <= 0 or token_count <= self.max_token_threshold))
        except Exception as e:
            logger.error(f"Error during token filtering: {e}")
            return False  # Filter out examples that cause errors

    def process(self) -> Set[str]:
        """
        Process the SWE-Gym dataset:
        1. Load dataset
        2. Clone/update repositories
        3. Extract file contents for each example
        4. Extract file statistics for all files in each repo
        5. Apply token count filtering
        6. Calculate simplified statistics
        7. Save processed dataset
        
        Returns:
            Set of unique repositories processed
        """
        # Check if processed data already exists
        if os.path.exists(self.dataset_path) and not self.cfg.swe.force_reprocess:
            logger.info(f"Found existing processed dataset at '{self.dataset_path}'. Attempting to load from disk.")
            try:
                processed_dataset = load_from_disk(self.dataset_path)
                logger.info("Processed dataset loaded successfully.")
                # Extract unique repository names from the processed dataset
                return set(processed_dataset['repo'])
            except Exception as e:
                logger.error(f"Failed to load dataset from '{self.dataset_path}': {e}")
                logger.warning("Proceeding with reprocessing the dataset.")

        # Load the original dataset from Hugging Face
        logger.info("Loading SWE-Gym dataset from Hugging Face...")
        dataset = load_dataset(self.cfg.swe.hf_dataset_name, split=self.cfg.swe.hf_split_name)
        logger.info("Dataset loaded.")
        logger.info(f"Original dataset size: {len(dataset)} examples")

        # Extract unique repository names from the 'repo' column
        unique_repos = set(dataset['repo'])
        logger.info(f"Found {len(unique_repos)} unique repositories to process.")

        # Clone or update repositories
        repo_paths = self.repo_manager.clone_or_update_repos(list(unique_repos))
        
        # Apply mapping function to extract file contents and file stats
        logger.info("Starting dataset mapping to extract file contents and compute file statistics...")
        processed_dataset = dataset.map(
            lambda example: self._process_example(example, repo_paths),
            batched=False,  # Process one example at a time
            num_proc=self.num_map_processes,
            load_from_cache_file=False,  # Ensure the mapping function runs
            desc="Extracting Gold File Contents and Computing File Stats"
        )
        logger.info("Dataset mapping finished.")
        
        # Apply token count filtering
        original_size = len(processed_dataset)
        logger.info(f"Filtering dataset by token count (min: {self.min_token_threshold}, max: {self.max_token_threshold if self.max_token_threshold > 0 else 'unlimited'})...")
        
        # Only filter if we have a tokenizer
        if self.tokenizer:
            filtered_dataset = processed_dataset.filter(
                self._filter_by_token_count,
                desc="Filtering by Token Count"
            )
            filtered_size = len(filtered_dataset)
            filtered_out = original_size - filtered_size
            
            logger.info(f"Token filtering complete. {filtered_out} examples filtered out ({filtered_size} remaining).")
            processed_dataset = filtered_dataset
        else:
            logger.warning("Tokenizer not available, skipping token-based filtering.")

        # Save the processed dataset
        logger.info(f"Saving processed dataset to '{self.dataset_path}'...")
        processed_dataset.save_to_disk(self.dataset_path)
        logger.info("Dataset saved successfully.")

        # Calculate and display simplified statistics
        self._calculate_simplified_statistics(processed_dataset, original_size)
        
        return unique_repos

    def _calculate_simplified_statistics(self, dataset, original_size):
        """
        Calculate and log simplified statistics about the processed dataset,
        focusing on filtering results and file statistics.
        
        Args:
            dataset: The processed dataset
            original_size: Size of the dataset before filtering
        """
        logger.info("\n--- Dataset Filtering Stats ---")
        logger.info(f"Original dataset size: {original_size} examples")
        logger.info(f"Filtered dataset size: {len(dataset)} examples")
        logger.info(f"Examples filtered out: {original_size - len(dataset)} examples")
        
        if self.tokenizer:
            logger.info(f"Token threshold criteria: min={self.min_token_threshold}, max={self.max_token_threshold if self.max_token_threshold > 0 else 'unlimited'}")
        else:
            logger.info("Note: Token-based filtering was skipped because tokenizer could not be loaded.")
        
        # Count examples with no content and file stats
        empty_content_count = 0
        empty_file_stats_count = 0
        total_files_across_examples = 0
        
        for example in tqdm(dataset, desc="Checking Content and File Stats"):
            try:
                file_contents_dict = json.loads(example.get('gold_file_contents', '{}'))
                if not file_contents_dict:
                    empty_content_count += 1
            except:
                empty_content_count += 1
                
            try:
                all_file_stats = json.loads(example.get('all_file_stats', '{}'))
                if not all_file_stats:
                    empty_file_stats_count += 1
                else:
                    total_files_across_examples += len(all_file_stats)
            except:
                empty_file_stats_count += 1
                
        avg_files_per_example = total_files_across_examples / len(dataset) if len(dataset) > 0 else 0
        
        logger.info(f"Examples with no gold file contents: {empty_content_count}")
        logger.info(f"Examples with gold file contents: {len(dataset) - empty_content_count}")
        logger.info(f"Examples with no file stats: {empty_file_stats_count}")
        logger.info(f"Examples with file stats: {len(dataset) - empty_file_stats_count}")
        logger.info(f"Average files per example: {avg_files_per_example:.1f}")
        logger.info(f"Total files processed across all examples: {total_files_across_examples}")
        logger.info("Statistics calculation completed.")


@hydra.main(config_path="../../conf", config_name="swe/preprocessor", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Process the SWE-Gym dataset using Hydra configuration.
    
    Args:
        cfg: Hydra configuration
    """
    # Process the dataset
    processor = SwePreprocessor(cfg)
    unique_repos = processor.process()
    
    logger.info(f"Processing completed successfully. Processed {len(unique_repos)} repositories.")
    return unique_repos


if __name__ == "__main__":
    main()