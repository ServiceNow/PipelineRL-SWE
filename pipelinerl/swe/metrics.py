"""
Updated unified metrics class for the SWE pipeline with generic self-evaluation and A2A tracking.
"""

from typing import Optional
from pipelinerl.rollouts import BaseMetrics


class UnifiedMetrics(BaseMetrics):
    """
    Updated metrics that track performance across pipeline stages with generic self-evaluation and A2A.
    """
    
    # === STAGE-SPECIFIC CORE METRICS ===
    # These reflect final performance (post-enhancement if A2A was used)
    
    # Localization metrics (core only)
    localization_mrr: Optional[float] = 0.0
    localization_ndcg: Optional[float] = 0.0       
    localization_num_queries: Optional[int] = 0
    localization_format_penalty: Optional[float] = 0.0
    localization_recall: Optional[float] = 0.0     

    # File selection metrics 
    selection_precision: Optional[float] = 0.0     
    selection_recall: Optional[float] = 0.0   
    selection_f1: Optional[float] = 0.0
    selection_format_penalty: Optional[float] = 0.0

    # Repair metrics
    repair_reward: Optional[float] = 0.0           
    repair_success: Optional[bool] = False         
    repair_format_error: Optional[bool] = False    

    # === PRE-ENHANCEMENT METRICS (for A2A comparison) ===
    # Only populated if A2A was actually triggered for that stage
    
    # Localization pre-enhancement
    localization_initial_mrr: Optional[float] = None
    localization_initial_recall: Optional[float] = None
    
    # File selection pre-enhancement
    selection_initial_precision: Optional[float] = None
    selection_initial_recall: Optional[float] = None
    selection_initial_f1: Optional[float] = None
    
    # Repair pre-enhancement
    repair_initial_reward: Optional[float] = None
    repair_initial_success: Optional[bool] = None

    # === GENERIC SELF-EVALUATION METRICS ===
    
    # Localization self-evaluation
    localization_self_eval_predicted_score: Optional[float] = 0.0
    localization_self_eval_prediction_error: Optional[float] = 1.0
    localization_self_eval_parsing_error: Optional[bool] = True
    
    # File selection self-evaluation  
    selection_self_eval_predicted_score: Optional[float] = 0.0
    selection_self_eval_prediction_error: Optional[float] = 1.0
    selection_self_eval_parsing_error: Optional[bool] = True
    
    # Repair self-evaluation
    repair_self_eval_predicted_score: Optional[float] = 0.0
    repair_self_eval_prediction_error: Optional[float] = 1.0
    repair_self_eval_parsing_error: Optional[bool] = True

    # === A2A TOKEN USAGE ===
    # Aggregated across all stages that used A2A
    
    # Query generation tokens
    total_a2a_query_prompt_tokens: Optional[int] = 0
    total_a2a_query_output_tokens: Optional[int] = 0
    
    # Expert consultation tokens
    total_a2a_expert_prompt_tokens: Optional[int] = 0
    total_a2a_expert_output_tokens: Optional[int] = 0

    # === PIPELINE-WIDE METRICS ===
    
    # Pipeline success metrics
    file_pipeline_success: Optional[bool] = False       
    total_pipeline_success: Optional[bool] = False      
    total_prompt_tokens: Optional[int] = 0
    total_output_tokens: Optional[int] = 0
    
    # Error tracking (per stage)
    localization_format_error: Optional[bool] = False
    selection_format_error: Optional[bool] = False
    repair_format_error: Optional[bool] = False
    
    # === ABSTENTION METRICS (CONFIG-DRIVEN) ===
    
    # These will be computed based on a fixed threshold from config
    # NOTE: This is ONLY for aggregate statistics, not for stopping the pipeline
    abstention_threshold: Optional[float] = 0.5  # Can be set from config
    
    # Override base class required fields
    reward: Optional[float] = 0.0                  
    success: Optional[bool] = False                
    no_error: Optional[bool] = True                
    no_answer: Optional[bool] = False              
    
    def set_abstention_threshold(self, threshold: float):
        """Set the abstention threshold from config."""
        self.abstention_threshold = threshold
    
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
        self.total_pipeline_success = (
            self.file_pipeline_success and 
            (self.repair_success or False)
        )
        
        # Error tracking
        self.localization_format_error = (self.localization_format_penalty or 0) > 0
        self.selection_format_error = (self.selection_format_penalty or 0) > 0
        
        self.no_error = not any([
            self.localization_format_error,
            self.selection_format_error, 
            self.repair_format_error or False,
            self.localization_self_eval_parsing_error or False,
            self.selection_self_eval_parsing_error or False,
            self.repair_self_eval_parsing_error or False
        ])