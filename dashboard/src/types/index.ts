export interface CausalAnalysisResult {
    ate: {
        value: number;
        ci_lower: number;
        ci_upper: number;
        alpha?: number;
        description?: string;
    };
    cate_distribution: {
        mean: number;
        std: number;
        min: number;
        max: number;
        histogram: {
            bin_edges: number[];
            counts: number[];
        };
    };
    segments: SegmentAnalysis[];
    dag: {
        nodes: DAGNode[];
        edges: DAGEdge[];
    };
    metadata: {
        generated_at: string;
        scenario: string;
        model_type: string;
        n_samples: number;
        feature_names: string[];
        treatment_col: string;
        outcome_col: string;
    };
    sensitivity: {
        status: string;
        placebo_test: { status: string; p_value: number; mean_effect?: number; null_mean?: number };
        random_common_cause: { status: string; stability: number; mean_effect: number };
        e_value?: { status: string; point: number; ci_bound: number; interpretation: string };
        overlap?: {
            status: string;
            overlap_score: number;
            interpretation: string;
            ps_stats?: {
                treated_mean: number; treated_std: number;
                control_mean: number; control_std: number;
            };
            iptw_max_weight?: number;
            pct_extreme_weights?: number;
            ps_histogram?: {
                bin_edges: number[];
                treated_counts: number[];
                control_counts: number[];
            };
        };
        gates?: {
            status: string;
            n_groups: number;
            f_statistic: number;
            heterogeneity: string;
            groups: {
                group_id: number;
                label: string;
                n: number;
                mean_cate: number;
                std_cate: number;
                ci_lower: number;
                ci_upper: number;
                clan_features: Record<string, number>;
            }[];
        };
    };
    explainability?: {
        feature_importance: { feature: string; importance: number }[];
        counterfactuals: {
            user_id: number;
            original_cate: number;
            counterfactual_cate: number;
            diff: number;
            description: string;
        }[];
    };
    estimation_accuracy?: {
        rmse: number;
        mae: number;
        bias: number;
        coverage_rate: number;
        correlation: number;
        n_samples: number;
    };
    ai_insights?: {
        summary: string;
        headline: string;
        significance: string;
        effect_size: string;
        effect_direction: string;
        top_drivers: { feature: string; importance: number }[];
        model_quality: string;
        model_quality_label: string;
        correlation: number;
        rmse: number;
        recommendation: string;
        generated_by: string;
    };
    // Phase 3 Extensions
    debate?: DebateResult;
    conformal_results?: ConformalResults;
    benchmark_results?: BenchmarkResults;
    scatter_data?: Record<string, number[]>;
}

export interface SegmentAnalysis {
    name: string;
    dimension: string;
    n: number;
    cate_mean: number;
    cate_ci_lower: number;
    cate_ci_upper: number;
}

export interface DAGNode {
    id: string;
    label: string;
    role: "treatment" | "outcome" | "confounder" | "mediator" | "other";
}

export interface DAGEdge {
    source: string;
    target: string;
}

// --- Phase 3 Interfaces ---

export interface DebateResult {
    verdict: "CAUSAL" | "NOT_CAUSAL" | "UNCERTAIN" | "UNKNOWN";
    confidence: number;
    pro_score: number;
    con_score: number;
    rounds: number;
    recommendation: string;
    pro_evidence: Evidence[];
    con_evidence: Evidence[];
}

export interface Evidence {
    claim: string;
    type: string;
    strength: number;
    source: string;
}

export interface ConformalResults {
    ci_lower_mean: number;
    ci_upper_mean: number;
    coverage: number;
    mode?: string;
}

export interface BenchmarkResults {
    [dataset: string]: {
        [method: string]: {
            pehe_mean: number;
            pehe_std: number;
            ate_bias_mean: number;
            ate_bias_std: number;
        };
    };
}
