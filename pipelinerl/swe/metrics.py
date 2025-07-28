"""
Unified metrics class for the SWE pipeline that combines localization, 
file selection, and repair metrics.
"""

from typing import Optional
from pipelinerl.rollouts import BaseMetrics


class UnifiedMetrics(BaseMetrics):
    """
    Unified metrics that can track performance across all pipeline stages.
    
    Fields are optional and will only be populated based on which stages
    are enabled in the configuration.
    """
    
    # Localization metrics (from localization_rollouts.py)
    localization_mrr: Optional[float] = None
    localization_ndcg_at_10: Optional[float] = None
    localization_num_queries: Optional[int] = None
    localization_total_query_length: Optional[int] = None
    localization_avg_query_length: Optional[float] = None
    localization_num_search_results: Optional[int] = None
    localization_format_violation: Optional[bool] = None
    localization_format_penalty: Optional[float] = None
    localization_garbage_length: Optional[int] = None
    localization_has_garbage_content: Optional[bool] = None
    localization_num_gold_files: Optional[int] = None
    localization_num_found: Optional[int] = None
    localization_best_rank: Optional[int] = None
    localization_worst_rank: Optional[int] = None
    localization_recall: Optional[float] = None
    localization_precision: Optional[float] = None
    localization_f1_score: Optional[float] = None
    
    # File selection metrics (new)
    selection_num_candidates: Optional[int] = None
    selection_num_selected: Optional[int] = None
    selection_accuracy: Optional[float] = None          # How many gold files were selected
    selection_precision: Optional[float] = None        # Selected relevant / total selected
    selection_recall: Optional[float] = None           # Selected relevant / total relevant
    selection_f1_score: Optional[float] = None         # F1 of selection precision/recall
    selection_format_violation: Optional[bool] = None
    selection_format_penalty: Optional[float] = None
    selection_garbage_length: Optional[int] = None
    selection_has_garbage_content: Optional[bool] = None
    
    # Repair metrics (from swe_rollouts.py)
    repair_reward: Optional[float] = None
    repair_success: Optional[bool] = None               # Repair reward > threshold
    repair_no_edits: Optional[bool] = None
    repair_num_files_changed: Optional[int] = None
    repair_format_error: Optional[bool] = None
    repair_avg_similarity: Optional[float] = None      # Same as repair_reward
    repair_oracle_files: Optional[int] = None
    repair_predicted_files: Optional[int] = None
    
    # Pipeline-wide metrics
    pipeline_success: Optional[bool] = None             # End-to-end success
    pipeline_total_steps: Optional[int] = None          # Number of stages that ran
    pipeline_completed_steps: Optional[int] = None     # Number of stages that succeeded
    
    # Token usage across all stages
    total_prompt_tokens: Optional[int] = None
    total_output_tokens: Optional[int] = None
    
    # Override the base class fields to make them optional for the unified case
    reward: Optional[float] = None                      # Primary reward (usually repair_reward)
    success: Optional[bool] = None                      # Primary success metric
    no_error: Optional[bool] = None                     # No errors across any stage
    no_answer: Optional[bool] = None                    # No valid output from any stage
    overflow: Optional[int] = None                      # Any stage hit length limits
    
    def compute_derived_metrics(self):
        """Compute derived metrics after all stages have completed."""
        
        # Set primary metrics based on repair if available, otherwise localization
        if self.repair_reward is not None:
            self.reward = self.repair_reward
            self.success = self.repair_success
        elif self.localization_mrr is not None:
            self.reward = self.localization_mrr
            self.success = self.localization_mrr > 0.5  # Arbitrary threshold
        
        # Compute pipeline success
        if self.repair_success is not None:
            self.pipeline_success = self.repair_success
        elif self.selection_accuracy is not None and self.selection_accuracy > 0.5:
            self.pipeline_success = True
        elif self.localization_recall is not None and self.localization_recall > 0.5:
            self.pipeline_success = True
        else:
            self.pipeline_success = False
        
        # Compute total tokens
        prompt_tokens = 0
        output_tokens = 0
        
        if hasattr(self, 'localization_prompt_tokens') and self.localization_prompt_tokens:
            prompt_tokens += self.localization_prompt_tokens
        if hasattr(self, 'localization_output_tokens') and self.localization_output_tokens:
            output_tokens += self.localization_output_tokens
            
        if hasattr(self, 'selection_prompt_tokens') and self.selection_prompt_tokens:
            prompt_tokens += self.selection_prompt_tokens
        if hasattr(self, 'selection_output_tokens') and self.selection_output_tokens:
            output_tokens += self.selection_output_tokens
            
        if hasattr(self, 'repair_prompt_tokens') and self.repair_prompt_tokens:
            prompt_tokens += self.repair_prompt_tokens
        if hasattr(self, 'repair_output_tokens') and self.repair_output_tokens:
            output_tokens += self.repair_output_tokens
        
        self.total_prompt_tokens = prompt_tokens if prompt_tokens > 0 else None
        self.total_output_tokens = output_tokens if output_tokens > 0 else None
        
        # Compute error flags
        self.no_error = not any([
            self.localization_format_violation,
            self.selection_format_violation,
            self.repair_format_error
        ])
        
        self.no_answer = any([
            self.localization_mrr == -1 if self.localization_mrr is not None else False,
            self.selection_accuracy == 0 if self.selection_accuracy is not None else False,
            self.repair_no_edits if self.repair_no_edits is not None else False
        ])
        
        # Compute pipeline step counts
        total_steps = 0
        completed_steps = 0
        
        if self.localization_mrr is not None:
            total_steps += 1
            if self.localization_mrr >= 0:  # -1 indicates failure
                completed_steps += 1
                
        if self.selection_accuracy is not None:
            total_steps += 1
            if self.selection_accuracy > 0:
                completed_steps += 1
                
        if self.repair_reward is not None:
            total_steps += 1
            if self.repair_reward >= 0:  # -1 indicates failure
                completed_steps += 1
        
        self.pipeline_total_steps = total_steps if total_steps > 0 else None
        self.pipeline_completed_steps = completed_steps if total_steps > 0 else None