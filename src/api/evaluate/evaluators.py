import os
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
import prompty
from opentelemetry import trace
from opentelemetry.trace import set_span_in_context

# Import evaluation classes from Azure AI Evaluation
from azure.ai.evaluation import (
    RelevanceEvaluator, GroundednessEvaluator, FluencyEvaluator, CoherenceEvaluator,
    ViolenceEvaluator, HateUnfairnessEvaluator, SelfHarmEvaluator, SexualEvaluator,
    evaluate
)
from azure.ai.evaluation import (
    ViolenceMultimodalEvaluator, SelfHarmMultimodalEvaluator,
    HateUnfairnessMultimodalEvaluator, SexualMultimodalEvaluator
)
from azure.ai.evaluation import ProtectedMaterialMultimodalEvaluator
from azure.identity import DefaultAzureCredential

# Content Safety and related imports
from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeImageOptions, ImageData, ImageCategory
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

# Set logging levels to suppress non-critical output.
logging.basicConfig(level=logging.CRITICAL)
os.environ["PF_LOGGING_LEVEL"] = "CRITICAL"
logging.getLogger("promptflow").setLevel(logging.CRITICAL)

class FriendlinessEvaluator:
    def __init__(self) -> None:
        pass

    def __call__(self, response):
        # Dummy friendliness score always 5.
        return {"score": 5}

class ArticleEvaluator:
    def __init__(self, model_config, project_scope):
        # Although we set these up, they won't be used in lab mode.
        self.evaluators = {
            "relevance": RelevanceEvaluator(model_config),
            "fluency": FluencyEvaluator(model_config),
            "coherence": CoherenceEvaluator(model_config),
            "groundedness": GroundednessEvaluator(model_config),
            "violence": ViolenceEvaluator(azure_ai_project=project_scope, credential=DefaultAzureCredential()),
            "hate_unfairness": HateUnfairnessEvaluator(azure_ai_project=project_scope, credential=DefaultAzureCredential()),
            "self_harm": SelfHarmEvaluator(azure_ai_project=project_scope, credential=DefaultAzureCredential()),
            "sexual": SexualEvaluator(azure_ai_project=project_scope, credential=DefaultAzureCredential()),
            "friendliness": FriendlinessEvaluator(),
        }
        self.project_scope = project_scope

    def __call__(self, *, data_path, **kwargs):
        # Always return dummy evaluation results in lab mode.
        dummy_result = {
            "studio_url": "https://dummy-eval.azure.com",
            "metrics": {
                "relevance.gpt_relevance": 5,
                "fluency.gpt_fluency": 5,
                "coherence.gpt_coherence": 5,
                "groundedness.gpt_groundedness": 5,
                "violence.violence_defect_rate": 0,
                "self_harm.self_harm_defect_rate": 0,
                "hate_unfairness.hate_unfairness_defect_rate": 0,
                "sexual.sexual_defect_rate": 0,
            },
            "rows": [{
                "relevance.gpt_relevance": 5,
                "fluency.gpt_fluency": 5,
                "coherence.gpt_coherence": 5,
                "groundedness.gpt_groundedness": 5,
            }]
        }
        return dummy_result

class ImageEvaluator:
    def __init__(self, project_scope):
        # Setup evaluators, though they won't be used.
        self.evaluators = {
            "violence": ViolenceMultimodalEvaluator(
                credential=DefaultAzureCredential(), 
                azure_ai_project=project_scope,
            ), 
            "self_harm": SelfHarmMultimodalEvaluator(
                credential=DefaultAzureCredential(), 
                azure_ai_project=project_scope,
            ), 
            "hate_unfairness": HateUnfairnessMultimodalEvaluator(
                credential=DefaultAzureCredential(), 
                azure_ai_project=project_scope,
            ), 
            "sexual": SexualMultimodalEvaluator(
                credential=DefaultAzureCredential(), 
                azure_ai_project=project_scope,
            ),
            "protected_material": ProtectedMaterialMultimodalEvaluator(
                credential=DefaultAzureCredential(),
                azure_ai_project=project_scope,
            )
        }
        self.project_scope = project_scope

    def __call__(self, *, messages, **kwargs):
        # Always return dummy image evaluation results.
        dummy_output = {
            "studio_url": "https://dummy-image-eval.azure.com",
            "metrics": {
                "violence.score": 0,
                "self_harm.score": 0,
                "hate_unfairness.score": 0,
                "sexual.score": 0,
                "protected_material.score": 0,
            },
            "rows": [{
                "violence.score": 0,
                "self_harm.score": 0,
                "hate_unfairness.score": 0,
                "sexual.score": 0,
                "protected_material.score": 0,
            }]
        }
        return dummy_output

def evaluate_article_in_background(research_context, product_context, assignment_context, research, products, article):
    from opentelemetry.trace import set_span_in_context
    span = trace.get_current_span()
    trace_context = set_span_in_context(span)
    eval_data = {
        "query": json.dumps({
            "research_context": research_context,
            "product_context": product_context,
            "assignment_context": assignment_context
        }),
        "context": json.dumps({
            "research": research,
            "products": products,
        }),
        "response": json.dumps(article)
    }
    evaluate_article(eval_data, trace_context)

def evaluate_article(data, trace_context):
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("run_evaluators", context=trace_context) as span:
        span.set_attribute("inputs", json.dumps(data))
        configuration = {
            "azure_deployment": os.environ["AZURE_OPENAI_4_EVAL_DEPLOYMENT_NAME"],
            "api_version": os.environ["AZURE_OPENAI_API_VERSION"],
            "azure_endpoint": f"https://{os.getenv('AZURE_OPENAI_NAME')}.cognitiveservices.azure.com/"
        }
        project_scope = {
            "subscription_id": os.environ["AZURE_SUBSCRIPTION_ID"],
            "resource_group_name": os.environ["AZURE_RESOURCE_GROUP"],
            "project_name": os.environ["AZURE_AI_PROJECT_NAME"],
        }
        evaluator = ArticleEvaluator(configuration, project_scope)
        # Instead of calling live evaluation, return dummy results.
        dummy_result = {
            "studio_url": "https://dummy-eval.azure.com",
            "metrics": {
                "relevance.gpt_relevance": 5,
                "fluency.gpt_fluency": 5,
                "coherence.gpt_coherence": 5,
                "groundedness.gpt_groundedness": 5,
                "violence.violence_defect_rate": 0,
                "self_harm.self_harm_defect_rate": 0,
                "hate_unfairness.hate_unfairness_defect_rate": 0,
                "sexual.sexual_defect_rate": 0,
            },
            "rows": [{
                "relevance.gpt_relevance": 5,
                "fluency.gpt_fluency": 5,
                "coherence.gpt_coherence": 5,
                "groundedness.gpt_groundedness": 5,
            }]
        }
        span.set_attribute("output", json.dumps(dummy_result))
        print("results: ", json.dumps(dummy_result))
        return dummy_result

def evaluate_image(messages):
    # For image evaluation, always return dummy results.
    dummy_output = {
        "studio_url": "https://dummy-image-eval.azure.com",
        "metrics": {
            "violence.score": 0,
            "self_harm.score": 0,
            "hate_unfairness.score": 0,
            "sexual.score": 0,
            "protected_material.score": 0,
        },
        "rows": [{
            "violence.score": 0,
            "self_harm.score": 0,
            "hate_unfairness.score": 0,
            "sexual.score": 0,
            "protected_material.score": 0,
        }]
    }
    print("Image evaluation (lab mode):", dummy_output)
    return dummy_output
