"""Analysis utilities for conversation metrics and context transformations."""

from .interface import (
    AggregatedMetrics,
    AnalysisResult,
    SectionMetrics,
    TokenizedConversation,
    TokenSection,
    TransformResult,
    aggregate_results,
    analyze_conversation,
    apply_context_transform,
    batch_tokenize_conversations,
    create_summarize_transform,
    extract_conversations,
    load_evaluation_results,
    save_token_plot_data_csv,
    save_transform_log_csv,
    tokenize_conversation,
    tokenize_conversation_simple,
)

from .uid_tracking import (
    ConversationWithUID,
    TransformRecord,
    apply_transform_with_tracking,
    assign_message_uids,
    config_to_string,
    get_message_uid,
    get_original_uid,
    get_transform_type_from_msg,
    parse_context_config,
)

from .oss_tokenizer import (
    batch_tokenize_with_sections,
    tokenize_conversation_oss,
    tokenize_conversation_with_sections,
)

from .summary_cache import (
    SummaryCache,
    SummaryEntry,
    apply_cached_transforms,
    apply_context_transformations,
    build_summary_cache,
    get_cache_key,
    get_cache_path,
    get_or_build_summary_cache,
    load_summary_cache,
    needs_summarization,
    save_summary_cache,
)

__all__ = [
    # Interface
    "AggregatedMetrics",
    "AnalysisResult",
    "SectionMetrics",
    "TokenizedConversation",
    "TokenSection",
    "TransformResult",
    "aggregate_results",
    "analyze_conversation",
    "apply_context_transform",
    "batch_tokenize_conversations",
    "create_summarize_transform",
    "extract_conversations",
    "load_evaluation_results",
    "save_token_plot_data_csv",
    "save_transform_log_csv",
    "tokenize_conversation",
    "tokenize_conversation_simple",
    # UID Tracking
    "ConversationWithUID",
    "TransformRecord",
    "apply_transform_with_tracking",
    "assign_message_uids",
    "config_to_string",
    "get_message_uid",
    "get_original_uid",
    "get_transform_type_from_msg",
    "parse_context_config",
    # OSS Tokenizer
    "batch_tokenize_with_sections",
    "tokenize_conversation_oss",
    "tokenize_conversation_with_sections",
    # Summary Cache
    "SummaryCache",
    "SummaryEntry",
    "apply_cached_transforms",
    "apply_context_transformations",
    "build_summary_cache",
    "get_cache_key",
    "get_cache_path",
    "get_or_build_summary_cache",
    "load_summary_cache",
    "needs_summarization",
    "save_summary_cache",
]
