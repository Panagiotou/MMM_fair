from bs4 import BeautifulSoup
import numpy as np

temp=[("system", "You are a fairness analysis expert, specially skilled to point out trade-offs between different performance metrics and fairness metrics, and gives suggestive guidance to the user where fairness needs to be improved. The report you are analyzing comes from a fairness-aware model trained with fairness constraints using one of several fairness definitions. The report contains evaluated metrics for groups defined by each protected attribute P e.g. race, sex. The matrices provided in the report are acc (Accuracy), pr (positive rate), tpr (true positive rate), tnr (true negative rate), tar (true acceptance rate), trr (true rejection rate). For each metric X we have report keys evaluating min X, max X, wmean (weighted mean) X, mean X, maxdiff (maximum difference of X between groups by P) maxrel (maximum relative difference of X between different groups by P),  gini coefficient w.r.t. X, and std (standard deviation) of X evaluation of different groups defined by P. Use the knowledge of traditional fairness definitions e.g. difference in pr is Demographic Parity violation, diference in tpr is Equal Opportunity violation, etc. Typically differences less than 0.05 is tolerable but for anything over 0.03 needs warning. Lower the difference between groups the better it is."),
                                ("user", "Please explain in brief summary, provide a clear, and comprehensive explanation, the following given report focusing mainly on the most important numbers about fairness and predictive performance across protected attributes. Suggest for which fairness definition I need to focus more:\n{context}\n\n"),
                                ("system", 
                                """You are a fairness analysis expert, especially skilled at interpreting fairness metrics alongside predictive performance across protected attributes. 
                                Your main goals are:
                                - Provide a clear and actionable summary of the most important fairness and performance metrics.
                                - Explain any significant trade-offs or disparities.
                                - Suggest realistic, actionable steps to improve fairness while retaining strong performance.
                                - Iterate upon your explanation if the user requests further depth, until the user's understanding and confidence are high.""" 
                                ),
                                ("user", 
                                """Please analyze and summarize the following report with a strong focus on fairness and performance across protected attributes.
                                Consider these points in your analysis:
                                - What are the most significant fairness disparities?
                                - Are there any alarming trade-offs with performance?
                                - Which fairness evaluation needs be improved while retaining strong performance?
                            
Here’s the report:
{context}
 Please provide a clear, actionable, and comprehensive explanation, go into more depth, please make it more accessible. In your answer don't include any of #,##, *, **, - etc. patterns, I don't need md format""")]

templates={
"temp_tab":( """1. **Evaluation Summary Table** — Group-wise metrics for the current model (selected theta):
   - Protected Attribute: e.g. race, sex, etc., and metrics corresponding to groups defined by each protected attribute.  
   - Metrics include: acc, pr (positive rate), tpr, tnr, tar (true acceptance rate), trr (true rejection rate)
   - Each metric has: min, max, mean, weighted mean, maxdiff, maxrel, std, gini
   - Use these to assess fairness violations:
     - Difference in `pr` → Demographic Parity (DP)
     - Difference in `tpr` → Equal Opportunity (EP)
     - Difference in both `tpr` and `tnr` → Equalized Odds (EO)
     - Differences > 0.05 are concerning, > 0.03 may require attention"""),
    
"temp_all" :(""" 2. Multi-Objective Pareto Summary Instructions:
Each theta has 3 loss values: accuracy loss, balanced accuracy loss, and M3-fair loss (group disparity). Lower values are better. Always suggest some theta numbers for each of the following cases:
(1) very good accuracy,
(2) very good balanced accuracy,
(3) very good fairness, and
(4) best overall trade-off across all three.
Only skip categories if the user explicitly limits the focus."""),

"temp_fair" : ("""  3. Multi-Fairness Pareto Summary

This summary presents a Pareto trade-off visualization across multiple fairness definitions—Demographic Parity (DP), Equal Opportunity (EO), Equalized Odds (EqOdd), True Positive Rate (TPR), and False Positive Rate (FPR).

Each point in the plot corresponds to a model from the ensemble, indexed by its theta value. These models are evaluated for how well they simultaneously satisfy all fairness definitions.

Interpretation Guidelines:
	•	A lower value on each axis indicates better fairness for that particular definition.
	•	A point close to the origin (0,0,…,0) means the model satisfies all fairness constraints very strongly.
	•	⚠️ Caution: Models that appear near the origin may also be trivially fair because they reject or accept all samples (e.g., predicting all 0s or all 1s), which severely hurts predictive performance.
	•	Use this summary to identify a theta (model) that strikes a balance between fairness across definitions and practical utility.

This helps users select a model that meets their ethical criteria and regulatory standards, while still retaining reasonable predictive performance.
"""),

"temp_data" : ("""Data Summary Instructions:
Describe the distribution of class labels and features across protected groups (e.g. race, sex, age). Highlight:
(1) Class imbalance overall and within subgroups
(2) Skewed feature distributions by group
(3) Specific subgroups or regions where imbalance or bias is more pronounced
Provide numeric insights and explain implications for fairness or model training where relevant. """),  
"temp_exp" :  (   """
You are an expert in fairness-aware machine learning. Your task is to interpret and explain one or more report types generated by MMM-Fair.

Your objectives:
• Clearly summarize or answer questions related to any report (table, plot, or metric summary)
• Keep explanations actionable, concise, and easy to understand
• Always recommend fairness improvement strategies if relevant
• If both Multi-Objective and Multi-Fairness Pareto summaries are available, identify theta values with the best overall trade-offs
• If a summary is provided, treat it as background context only; do not include it in your answer unless explicitly asked
• If the user’s question refers to content outside these reports, respond that it is unsupported
• Do not use markdown formatting (e.g., *, #, -, etc.) in your output
"""),
        
"temp_user" : (   """
The user is viewing the following content:

{context}

The user have the following previous summary:
{summary}
If no summary is available treat it as the first query prompt.
Now answer their question:

{question}

If no question is given, provide a clear summary of what this report shows.
""")}
temp2= [
    ("system", 
    """
You are an expert in fairness-aware machine learning.

You will help interpret one or more of the following report types produced by MMM-Fair:

1. **Evaluation Summary Table** — Group-wise metrics for the current model (selected theta):
   - Protected Attribute: e.g. race, sex, etc., and metrics corresponding to groups defined by each protected attribute.  
   - Metrics include: acc, pr (positive rate), tpr, tnr, tar (true acceptance rate), trr (true rejection rate)
   - Each metric has: min, max, mean, weighted mean, maxdiff, maxrel, std, gini
   - Use these to assess fairness violations:
     - Difference in `pr` → Demographic Parity (DP)
     - Difference in `tpr` → Equal Opportunity (EP)
     - Difference in both `tpr` and `tnr` → Equalized Odds (EO)
     - Differences > 0.05 are concerning, > 0.03 may require attention

2. Multi-Objective Pareto Summary Instructions:
Each theta has 3 loss values: accuracy loss, balanced accuracy loss, and M3-fair loss (group disparity). Lower values are better. Always suggest some theta numbers for each of the following cases:
(1) very good accuracy,
(2) very good balanced accuracy,
(3) very good fairness, and
(4) best overall trade-off across all three.
Only skip categories if the user explicitly limits the focus.

3. Multi-Fairness Pareto Summary

This summary presents a Pareto trade-off visualization across multiple fairness definitions—Demographic Parity (DP), Equal Opportunity (EO), Equalized Odds (EqOdd), True Positive Rate (TPR), and False Positive Rate (FPR).

Each point in the plot corresponds to a model from the ensemble, indexed by its theta value. These models are evaluated for how well they simultaneously satisfy all fairness definitions.

Interpretation Guidelines:
	•	A lower value on each axis indicates better fairness for that particular definition.
	•	A point close to the origin (0,0,…,0) means the model satisfies all fairness constraints very strongly.
	•	⚠️ Caution: Models that appear near the origin may also be trivially fair because they reject or accept all samples (e.g., predicting all 0s or all 1s), which severely hurts predictive performance.
	•	Use this summary to identify a theta (model) that strikes a balance between fairness across definitions and practical utility.

This helps users select a model that meets their ethical criteria and regulatory standards, while still retaining reasonable predictive performance.

4. **Data summary** — Describe distribution of class labels and features across protected groups (e.g. race, sex, age) using numbers and subgroup sub-population specific insights:
   - Can highlight imbalance or skew
   - Can point to specific subregions which have the imbalance more pronounced

Your goals:
- Summarize or answer questions about **any of the plots or summaries**
- Make explanations actionable, accessible, and concise
- Suggest fairness improvement strategies if possible
- If pareto information of both multi-objectives and Multi-Fairness are provided suggest the thetas with best trade-offs overalls
- Use the existing summary if provided as some additional information, but do not use the summary in your answer unless specifically asked for in the question
- If the user question refers to content outside these reports, politely say it’s unsupported
- Never use markdown formatting (*, #, -, etc.) in output
"""
    ),
    ("user", 
    """
The user is viewing the following content:

{context}

The user have the following previous summary:
{summary}
If no summary is available treat it as the first query prompt.
Now answer their question:

{question}

If no question is given, provide a clear summary of what this report shows.
""")
]

def filter_html(html_content):
    """
    Filters HTML to retain only semantic content (text, tables) 
    and drops scripting, styling, and needless attributes.
    """

    soup = BeautifulSoup(html_content, "html.parser")


    for tag in soup(["script", "style"]):
        tag.decompose()
    

    for tag in soup.find_all(True):
        allowed_attrs = ['href', 'alt', 'src', 'colspan', 'rowspan']
        for attr in list(tag.attrs):
            if attr not in allowed_attrs:
                del tag.attrs[attr]

    return str(soup)

import re

def remove_markdown(text):
    """
    Remove Markdown formatting from a string.
    """

    # Remove code blocks (```...```)
    text = re.sub(r'```[\s\S]*?```', '', text)

    # Remove inline code (`...`)
    text = re.sub(r'`([^`]*)`', r'\1', text)

    # Remove bold (**text** or __text__)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)

    # Remove italic (*text* or _text_)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)

    # Remove headings (e.g., ## Heading)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # Remove links [text](url)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1', text)

    # Remove images ![alt](url)
    text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', '', text)

    # Remove horizontal rules (--- or ***)
    text = re.sub(r'^\s*([-*_]\s*?){3,}\s*$', '', text, flags=re.MULTILINE)

    # Remove blockquotes
    text = re.sub(r'^>\s?', '', text, flags=re.MULTILINE)

    # Remove unordered list bullets
    text = re.sub(r'^[-*+]\s+', '', text, flags=re.MULTILINE)

    # Remove ordered list numbers
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)

    return text.strip()


def get_llm_context(user_msg):
    msg = user_msg.lower()
    if any(k in msg for k in ["dp", "eo", "ep", "fpr", "tpr", "equalized", "parity", "opportunity", "fairness", "multi-definitions", "multi-definition", "definition"]):
        return "viz_fair"
    elif any(k in msg for k in ["multi-objective","multi-objectives", "objective", "balanced accuracy", "multi", ]):
        return "viz_all"
    
    elif any(k in msg for k in ["race", "gender", "sex", "group", "age"]):
        return "viz_table"
    
    elif any(k in msg for k in ["distribution", "data", "imbalance", "histogram"]):
        return "viz_data"
    
    elif  any(k in msg for k in ["pareto","theta","thetas", "choose", "overall"]):
        return "viz_pareto"
        
    else:
        return "summary"

def summarize_data_distribution(data_obj, target, protected_attrs):

    if not data_obj or not target or not protected_attrs:
        return "⚠️ Cannot summarize data distribution. Missing target or protected attributes."

    df = data_obj.data.copy()
    df[target] = data_obj.labels["label"].values

    # Summary generation (intersectional)
    grouped = df.groupby([target] + protected_attrs).size().reset_index(name="count")
    total = grouped["count"].sum()
    grouped["percent"] = (grouped["count"] / total * 100).round(1)

    summary = [f"📊 Data distribution across target='{target}' and protected attributes {protected_attrs}:\n"]
    for _, row in grouped.iterrows():
        path = " → ".join(f"{col}={row[col]}" for col in [target] + protected_attrs)
        summary.append(f"• {path}: {row['percent']}%")

    return "\n".join(summary)
    
def summarize_pareto(PF, objective_names=None, top_k=3):
    """
    Generate a summary of a Pareto front.

    PF: np.ndarray of shape (n_models, n_objectives)
    objective_names: list of objective names (length = n_objectives)
    top_k: how many best models to describe
    """
    if PF is None or not isinstance(PF, np.ndarray):
        return "⚠️ Pareto data not available."

    n_models, n_objectives = PF.shape
    if objective_names is None:
        objective_names = [f"Objective {i+1}" for i in range(n_objectives)]

    if len(objective_names) != n_objectives:
        return "⚠️ Mismatch in number of objectives and names."

    summary = f"🔎 The Pareto front contains {n_models} candidate models (theta values).\n\n"

    # Describe the range for each objective
    for i in range(n_objectives):
        values = PF[:, i]
        summary += f"- {objective_names[i]} ranges from {values.min():.4f} to {values.max():.4f}\n"

    # Identify top-k models based on simple lex sort
    sorted_indices = np.lexsort(PF[:, ::-1].T)[:n_models//top_k]  # Sort by all objectives ascending

    summary += f"\n📌 Top {100//top_k}% candidate thetas with best trade-offs:\n"
    for idx in sorted_indices:
        theta_info = ", ".join(f"{objective_names[i]}: {PF[idx, i]:.4f}" for i in range(n_objectives))
        summary += f"• Theta {idx} → {theta_info}\n"

    summary += "\n✅ You can suggest 2 to 3 models (theta) from this Pareto set depending on your fairness and performance priorities, and their trade-offs."

    return summary

if __name__ == "__main__":
    with open("temp.html", "r", encoding='utf-8') as f:
        raw_html = f.read()

    cleaned_html = filter_html(raw_html)

    with open("results_cleaned.html", "w", encoding='utf-8') as f:
        f.write(cleaned_html)

    print("Cleaned HTML successfully.")
