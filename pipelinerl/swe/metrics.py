"""
Streamlined unified metrics class for the SWE pipeline.
"""

from typing import Optional
from pipelinerl.rollouts import BaseMetrics


class UnifiedMetrics(BaseMetrics):
    """
    Streamlined metrics that track performance across pipeline stages.
    Focuses on the most important metrics while reducing redundancy.
    """
    
    # === STAGE-SPECIFIC CORE METRICS ===
    
    # Localization metrics (core only)
    localization_mrr: Optional[float] = 0.0
    localization_ndcg: Optional[float] = 0.0       # For comparability with other systems
    localization_num_queries: Optional[int] = 0
    localization_format_penalty: Optional[float] = 0.0
    localization_recall: Optional[float] = 0.0     # What % of gold files found

    # File selection metrics 
    selection_precision: Optional[float] = 0.0     # Selected relevant / total selected
    selection_recall: Optional[float] = 0.0        # Selected relevant / total relevant  
    selection_format_penalty: Optional[float] = 0.0

    # Repair metrics
    repair_reward: Optional[float] = 0.0           # Patch similarity score
    repair_success: Optional[bool] = False         # Reward > threshold
    repair_format_error: Optional[bool] = False    # Failed to parse edits

    # Self-evaluation metrics
    self_eval_predicted_score: Optional[float] = 0.0    # Predicted repair quality (0.0-1.0)
    self_eval_prediction_error: Optional[float] = 1.0   # |predicted - actual|
    self_eval_reward: Optional[float] = 0.0             # Reward for prediction accuracy
    self_eval_success: Optional[bool] = False             # Prediction within threshold
    self_eval_parsing_error: Optional[bool] = True       # Failed to parse score

    # === PIPELINE-WIDE METRICS ===
    
    # Pipeline success metrics
    file_pipeline_success: Optional[bool] = False       # Perfect recall at localization AND selection
    total_pipeline_success: Optional[bool] = False      # File pipeline success AND repair success
    total_prompt_tokens: Optional[int] = 0
    total_output_tokens: Optional[int] = 0
    
    # Error tracking (per stage)
    localization_format_error: Optional[bool] = False
    selection_format_error: Optional[bool] = False
    repair_format_error: Optional[bool] = False
    
    # Override base class required fields
    reward: Optional[float] = 0.0                  # Primary reward (repair > localization)
    success: Optional[bool] = False                # Primary success metric
    no_error: Optional[bool] = True                # No errors across pipeline
    no_answer: Optional[bool] = False              # No answer provided by agent
    
    def compute_derived_metrics(self):
        """Compute derived metrics after all stages have completed."""
        
        # Primary metrics: repair takes precedence, then localization
        if self.repair_reward is not None:
            self.reward = self.repair_reward
            self.success = self.repair_success
        elif self.localization_mrr is not None:
            self.reward = self.localization_mrr
            self.success = self.localization_mrr > 0.3  # Reasonable threshold
        
        # File pipeline success = perfect recall at both localization and selection
        self.file_pipeline_success = (
            (self.localization_recall or 0) == 1.0 and 
            (self.selection_recall or 0) == 1.0
        )
        
        # Total pipeline success = file pipeline success + repair success
        # Note: Self-eval success is NOT included in pipeline success since it's evaluative, not generative
        self.total_pipeline_success = (
            self.file_pipeline_success and 
            (self.repair_success or False)
        )
        
        # Error tracking
        self.localization_format_error = (self.localization_format_penalty or 0) > 0
        self.selection_format_error = (self.selection_format_penalty or 0) > 0
        # repair_format_error and self_eval_parsing_error are set directly in their respective stages
        
        self.no_error = not any([
            self.localization_format_error,
            self.selection_format_error, 
            self.repair_format_error or False,
            self.self_eval_parsing_error or False
        ])