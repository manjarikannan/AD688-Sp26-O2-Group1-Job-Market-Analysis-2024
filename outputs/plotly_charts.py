from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lower, round as spark_round
import plotly.io as pio
import plotly.express as px

pio.templates.default = "plotly_white"

spark = SparkSession.builder \
    .appName("JobMarket") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

df_clean = spark.read.parquet("data/lightcast_clean.parquet")

ai_keywords = "ai|machine learning|artificial intelligence|deep learning|nlp|llm|data science|neural network"
df_clean = df_clean.withColumn(
    "IS_AI_ROLE",
    when(
        lower(col("TITLE_RAW")).rlike(ai_keywords) |
        lower(col("SKILLS_NAME")).rlike(ai_keywords),
        1
    ).otherwise(0)
)

# Chart 1: Salary by industry
industry_df = df_clean.groupBy("NAICS_2022_2_NAME", "IS_AI_ROLE") \
    .avg("SALARY_MID") \
    .withColumn("avg_salary", spark_round("avg(SALARY_MID)", 0)) \
    .filter(col("NAICS_2022_2_NAME").isNotNull()) \
    .toPandas()
industry_df.columns = ["industry", "is_ai", "avg_salary_raw", "avg_salary"]
industry_df["Role Type"] = industry_df["is_ai"].map({0: "Non-AI", 1: "AI"})

fig1 = px.bar(
    industry_df.sort_values("avg_salary"),
    x="avg_salary", y="industry", color="Role Type", barmode="group",
    title="Average Salary: AI vs Non-AI Roles by Industry",
    labels={"avg_salary": "Average Salary (USD)", "industry": "Industry"},
    color_discrete_map={"Non-AI": "#4C72B0", "AI": "#DD8452"}
)
fig1.update_layout(font=dict(family="Roboto", size=12))
fig1.write_image("outputs/salary_by_industry_plotly.png", width=1000, height=700)
print("Chart 1 saved.")

# Chart 2: Salary distribution
salary_df = df_clean.select("SALARY_MID", "IS_AI_ROLE") \
    .filter(col("SALARY_MID").isNotNull()) \
    .toPandas()
salary_df["Role Type"] = salary_df["IS_AI_ROLE"].map({0: "Non-AI", 1: "AI"})

fig2 = px.histogram(
    salary_df, x="SALARY_MID", color="Role Type", barmode="overlay",
    nbins=50, opacity=0.7,
    title="Salary Distribution: AI vs Non-AI Roles",
    labels={"SALARY_MID": "Salary (USD)"},
    color_discrete_map={"Non-AI": "#4C72B0", "AI": "#DD8452"}
)
fig2.update_layout(font=dict(family="Roboto", size=12))
fig2.write_image("outputs/salary_distribution_plotly.png", width=1000, height=600)
print("Chart 2 saved.")

# Chart 3: Top industries by AI job count
from pyspark.sql.functions import desc
top_industries = df_clean.filter(col("IS_AI_ROLE") == 1) \
    .filter(col("NAICS_2022_2_NAME").isNotNull()) \
    .groupBy("NAICS_2022_2_NAME").count() \
    .orderBy(desc("count")).limit(10).toPandas()

fig3 = px.bar(
    top_industries.sort_values("count"),
    x="count", y="NAICS_2022_2_NAME",
    title="Top Industries by AI Job Posting Count",
    labels={"count": "Number of Postings", "NAICS_2022_2_NAME": "Industry"},
    color_discrete_sequence=["#4C72B0"]
)
fig3.update_layout(font=dict(family="Roboto", size=12))
fig3.write_image("outputs/top_industries_plotly.png", width=1000, height=600)
print("Chart 3 saved.")

print("All charts done.")
