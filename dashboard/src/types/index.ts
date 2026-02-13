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
        placebo_test: { status: string; p_value: number; mean_effect: number };
        random_common_cause: { status: string; stability: number; mean_effect: number };
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
