####################################################################################
# App Name: ScreenLoom
# Date Modified: 19/05/2025
# version: V1.5
####################################################################################

import os
import streamlit as st

from utils.topics import TopicExtractor, TopicReducer
from utils.association import Associator
from utils.summary import SummaryGenerator
from utils.utils import time_execution, logger
from utils.connectors import load_file_data, get_excel_sheets,\
            get_sqlite_tables, load_sqlite_data,\
            get_db_tables, load_db_data

import pandas as pd
import time
import random
random.seed(42)

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

if not load_dotenv():
    st.warning("Please setup environment file before running this script.")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Environment not correctly initialized. Please add variable 'GROQ_API_KEY'.")
    st.stop()

TOPIC_EXTRACTOR_CONFIG = {
    "EMBEDDER_CHARACTER_LIMIT": 255,
    "TOPIC_GENERATOR_CHARACTER_LIMIT": 10_000
}

logger.info("App Started")

def mock_running(status_bar, timer: int, text: str):
    """Simulates a running process with a progress bar for a given duration.

    Args:
        status_bar: Streamlit progress bar object to update.
        timer (int): Total time in seconds to simulate the process.
        text (str): Text to display in the progress bar.
    """
    for i in range(100):
        time.sleep(random.randint(0, 2) * (timer / 100))
        status_bar.progress(i, text)

@time_execution
@st.cache_data(show_spinner=False)
def extract_topics(df: pd.DataFrame, selected_column: str) -> pd.DataFrame:
    """Extracts topics from a DataFrame column using an LLM-based TopicExtractor.

    Args:
        df (pd.DataFrame): Input DataFrame containing the text data.
        selected_column (str): Name of the column to extract topics from.

    Returns:
        pd.DataFrame: DataFrame containing extracted topics.

    Raises:
        Exception: If topic extraction fails, falls back to cached data.
    """
    status_bar = st.progress(0, text="Extracting topics...")
    try:
        raise Exception("testing")  # Temporary testing exception
        te = TopicExtractor(data=df, selected_column=selected_column,
                            status_bar=status_bar, config=TOPIC_EXTRACTOR_CONFIG)
        topics = te.fit_transform(llm_api_key=GROQ_API_KEY)
    except Exception as e:
        logger.critical("Error %s with extract_topics, defaulting to cache.", e)
        mock_running(status_bar, 15, "Extracting topics...")
        topics = pd.read_csv("static/medical/topics.csv")
    status_bar.empty()
    return topics

@time_execution
@st.cache_data(show_spinner=False)
def reduce_topics(df: pd.DataFrame) -> pd.DataFrame:
    """Reduces the number of topics in a DataFrame using TopicReducer.

    Args:
        df (pd.DataFrame): DataFrame with topics to reduce.

    Returns:
        pd.DataFrame: DataFrame with reduced topics.

    Raises:
        Exception: If topic reduction fails, falls back to cached data.
    """
    status_bar = st.progress(0, text="Reducing topics...")
    try:
        raise Exception("testing")  # Temporary testing exception
        tr = TopicReducer(df, status_bar)
        reduced_topics = tr.fit_transform()
    except Exception as e:
        logger.critical("Error %s with reduce_topics, defaulting to cache.", e)
        mock_running(status_bar, 10, "Reducing topics...")
        reduced_topics = pd.read_csv("static/medical/reduced_topics.csv")
    status_bar.empty()
    return reduced_topics

@time_execution
@st.cache_data(show_spinner=False)
def extract_features(df: pd.DataFrame,
                     topics_df: pd.DataFrame,
                     selected_column: str,
                     entity_column: str
                     ) -> pd.DataFrame:
    """Extracts features and associations from text data based on topics.

    Args:
        df (pd.DataFrame): Original DataFrame with text data.
        topics_df (pd.DataFrame): DataFrame with extracted topics.
        selected_column (str): Name of the column to analyze.
        entity_column (str): Name of the entity column (e.g., drug names).

    Returns:
        pd.DataFrame: DataFrame with extracted features and associations.

    Raises:
        Exception: If feature extraction fails, falls back to cached data.
    """
    status_bar = st.progress(0, text="Extracting features...")
    try:
        raise Exception("testing")  # Temporary testing exception
        a = Associator(data=df, topics_df=topics_df, selected_column=selected_column,
                       entity_column=entity_column, status_bar=status_bar)
        associations = a.fit_transform(llm_api_key=GROQ_API_KEY)
    except Exception as e:
        logger.critical("Error %s with extract_features, defaulting to cache.", e)
        mock_running(status_bar, 20, "Extracting features...")
        associations = pd.read_csv("static/medical/features.csv")
        associations['issues'] = associations['issues'].str.split(',', expand=False)
    status_bar.empty()
    return associations

@time_execution
def create_recommendation_table(associations_df: pd.DataFrame, topic_column: str):
    """Creates a recommendation table from associations, focusing on negative sentiment.

    Args:
        associations_df (pd.DataFrame): DataFrame with associations and features.
        topic_column (str): Column name for topics or entities.

    Returns:
        pd.DataFrame: Grouped DataFrame with recommendations sorted by sentiment strength.
    """
    eda_df = associations_df[[topic_column, 'sentiment', 'strength', 'issues']]
    eda_df = eda_df[eda_df['sentiment'] == "NEGATIVE"]
    grouped_df = eda_df.groupby(topic_column).agg({
        "strength": 'mean',
        "issues": list
    })
    grouped_df = grouped_df.sort_values(by='strength', ascending=False)

    return grouped_df

def create_topic_count_table(associations_df: pd.DataFrame, topic_column: str) -> pd.DataFrame:
    """Creates a grouped table of topic counts by sentiment for heatmap visualization.

    Args:
        associations_df (pd.DataFrame): DataFrame with associations and features.

    Returns:
        pd.DataFrame: Pivot table with topics as rows, sentiments as columns, and document
        counts as values.
    """
    topic_counts = associations_df.groupby([topic_column, 'sentiment']).size().reset_index(name='Count')
    pivot_df = topic_counts.pivot_table(
        index=topic_column,
        columns='sentiment',
        values='Count',
        fill_value=0
    ).reset_index()
    for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
        if sentiment not in pivot_df.columns:
            pivot_df[sentiment] = 0
    return pivot_df

# @st.cache_data(show_spinner=False)
@time_execution
def create_conclusion(topic_issues_pair):
    """Generates a summary conclusion from topic-issues pairs using SummaryGenerator.

    Args:
        topic_issues_pair: List of tuples pairing topics/entities with their issues.

    Returns:
        pd.DataFrame: DataFrame containing the generated summary.

    Raises:
        Exception: If summary generation fails, falls back to cached data (placeholder file needed).
    """
    status_bar = st.progress(0, "Generating summary...")
    try:
        sg = SummaryGenerator(issues_list=topic_issues_pair, status_bar=status_bar)
        summary = sg.fit_transform(api_key=GROQ_API_KEY)
    except Exception as e:
        logger.critical("Error %s with create_conclusion, defaulting to cache.", e.with_traceback)
        mock_running(status_bar, 20, "Generating summary...")
        with open("static/medical/summary.txt") as f:
            summary = f.read()
    status_bar.empty()
    return summary

def load_landing():
    """Displays the landing page with app introduction and features."""
    logger.info("Loaded landing page.")
    st.markdown("""
        <style>
        :root {
            --subtitle-color: #c1c9c9;
            --bg-secondary: #262730;
        }
        .title {
            font-size: 48px;
            font-weight: bold;
            color: #2c3e50;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            font-size: 20px;
            color: var(--subtitle-color);
            text-align: center;
            margin-bottom: 30px;
        }
        .feature-box {
            background-color: var(--bg-secondary);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            color: #95a5a6;
            margin-top: 50px;
        }
        </style>
                
        <script>
            // const isDarkMode = document.body.classList.contains('st-dark-theme');
            const isDarkMode = false;

            // Update CSS variables based on theme
            if (isDarkMode) {
                document.documentElement.style.setProperty('--bg-secondary', '#262730');
                document.documentElement.style.setProperty('--subtitle-color', '#c1c9c9');
            } else {
                document.documentElement.style.setProperty('--bg-secondary', '#F0F2F6');
                document.documentElement.style.setProperty('--subtitle-color', '#000000');
            }
        </script>
    """, unsafe_allow_html=True)

    st.image("static/logo_v3_dark.png", use_container_width=True)
    st.markdown(
        '<div class="subtitle">Unravel Insights with Topic Modeling & Text Analysis</div>',
        unsafe_allow_html=True
    )

    st.write("ScreenLoom leverages advanced Large Language Models (LLMs) to extract topics "
             "from text data, associate rows with topics, and uncover sentiments about "
             "pharmaceutical products. Empower your team to make data-driven decisions "
             "with precision and clarity.")

    st.subheader("Why ScreenLoom?")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="feature-box"><strong>Topic Extraction</strong><br>'
            'Automatically identify key topics from unstructured text using '
            'cutting-edge LLMs.</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="feature-box"><strong>Sentiment Analysis</strong><br>'
            'Gauge sentiments around your products to understand customer perceptions.</div>',
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            '<div class="feature-box"><strong>Row-to-Topic Mapping</strong><br>'
            'Link individual data rows to relevant topics for detailed insights.</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="feature-box"><strong>Actionable Insights</strong><br>'
            'Enable pharmaceutical teams to act confidently with clear, data-backed findings.</div>',
            unsafe_allow_html=True
        )

    st.header("Get Started")
    st.write("Connect to a data source to get started with your analysis!")

def load_guide():
    """Displays a concise user guide."""
    logger.info("Loaded user guide.")
    st.markdown("""
        <style>
        :root {
            --subtitle-color: #c1c9c9;
            --bg-secondary: #262730;
        }
        .title {
            font-size: 48px;
            font-weight: bold;
            color: #2c3e50;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            font-size: 20px;
            color: var(--subtitle-color);
            text-align: center;
            margin-bottom: 20px;
        }
        .feature-box {
            background-color: var(--bg-secondary);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            color: #95a5a6;
            margin-top: 20px;
        }
        </style>
                
        <script>
            const isDarkMode = false;
            if (isDarkMode) {
                document.documentElement.style.setProperty('--bg-secondary', '#262730');
                document.documentElement.style.setProperty('--subtitle-color', '#c1c9c9');
            } else {
                document.documentElement.style.setProperty('--bg-secondary', '#F0F2F6');
                document.documentElement.style.setProperty('--subtitle-color', '#000000');
            }
        </script>
    """, unsafe_allow_html=True)

    st.image("static/logo_v3_dark.png", use_container_width=True)
    st.markdown(
        '<div class="subtitle">Unravel Insights with Topic Modeling & Text Analysis</div>',
        unsafe_allow_html=True
    )

    st.write("ScreenLoom leverages advanced Large Language Models (LLMs) to extract topics "
             "from text data, associate rows with topics, and uncover sentiments about "
             "pharmaceutical products. Empower your team to make data-driven decisions "
             "with precision and clarity.")

    st.subheader("Why ScreenLoom?")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="feature-box"><strong>Topic Extraction</strong><br>'
            'Automatically identify key topics from unstructured text using '
            'cutting-edge LLMs.</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="feature-box"><strong>Sentiment Analysis</strong><br>'
            'Gauge sentiments around your products to understand customer perceptions.</div>',
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            '<div class="feature-box"><strong>Row-to-Topic Mapping</strong><br>'
            'Link individual data rows to relevant topics for detailed insights.</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="feature-box"><strong>Actionable Insights</strong><br>'
            'Enable pharmaceutical teams to act confidently with clear, data-backed findings.</div>',
            unsafe_allow_html=True
        )

    st.header("Connecting to Data")
    st.write("""
    1. **Select Source**: Choose **File** or **Server** in the sidebar.
    2. **File**: Upload CSV, Excel, JSON, text, or SQLite. Select sheet/table if needed. Click **Connect**.
    3. **Server**: Choose database (PostgreSQL, MySQL, SQL Server), enter credentials, select table, and click **Connect**.
    """)

    st.header("Column Selection")
    st.write("""
    - **Text Column**: Select a column with text (e.g., reviews) in the sidebar.
    - **Entity Column (Optional)**: Choose a column (e.g., drug names) for entity-specific analysis.
    """)

    st.header("Analysis Tabs")
    st.write("""
    Navigate using **Previous**, **Next**, **Disconnect** buttons.
    1. **Data Preview**: View dataset; verify data.
    2. **Topic Extraction**: Edit auto-extracted topics, approve changes.
    3. **Feature Extraction**: See topic/entity associations, sentiments, issues.
    4. **Recommendations**: View heatmap, bar chart, recommendation table, conclusion.
    """)

    st.header("Visualizing Results")
    st.write("""
    - **Heatmap**: Shows topic/entity sentiment distribution.
    - **Bar Chart**: Top 5 negative sentiment topics/entities.
    - **Recommendation Table**: Negative sentiment issues.
    - **Conclusion**: Summarizes actionable insights.
    """)

    st.header("Tips for Success")
    st.write("""
    - Use rich text columns; clear entity identifiers.
    - Verify file formats, database credentials.
    - Refine topics for accuracy.
    - Prioritize negative sentiments in **Recommendations**.
    - Ensure `.env` has `GROQ_API_KEY`.
    - Monitor progress bars for large datasets.
    """)

    st.markdown(
        '<div class="footer">Start uncovering insights with ScreenLoom!</div>',
        unsafe_allow_html=True
    )

def infer_file_type(filename):
    """
    Infer the file type from the file extension.
    
    Args:
        filename (str): Name of the uploaded file.
    
    Returns:
        str or None: Inferred file type ('csv', 'json', 'excel', 'text', 'sqlite') or None if unsupported.
    """
    extension = os.path.splitext(filename.lower())[1]
    extension_map = {
        '.csv': 'csv',
        '.json': 'json',
        '.xlsx': 'excel',
        '.xls': 'excel',
        '.txt': 'text',
        '.db': 'sqlite',
        '.sqlite': 'sqlite'
    }
    return extension_map.get(extension)

@st.dialog("User Guide", width='large')
def open_guide():
    """Opens the landing page in a dialog box."""
    load_guide()

if st.sidebar.button("User Guide"):
    open_guide()


if "df" not in st.session_state:
    st.session_state.df = None

def clear_variables():
    st.session_state.df = None
    st.session_state.active_tab = -1

source_type = st.sidebar.selectbox("Select Source:", [None, "File", "Server"], on_change=clear_variables)
if not source_type:
    load_landing()
    st.stop()

st.image("static/logo_v3_dark.png", use_container_width=True)

def file_upload(main_container):
    file = main_container.file_uploader("Please upload a file:", ['csv', 'xls', 'xlsx', 'txt', 'json', 'db', 'sqlite'])

    if not file:
        # load_landing()
        st.stop()

    file_type = infer_file_type(file.name)
    excel_sheet = None
    if file_type == "excel":
        file.seek(0)
        try:
            sheets = get_excel_sheets(file)
            excel_sheet = main_container.selectbox("Select a sheet:", sheets)
        except Exception as e:
            # load_landing()
            st.error(f"Error loading Excel sheets: {e}")
            st.stop()
    
    if file_type == "sqlite":
        file.seek(0)
        try:
            tables = get_sqlite_tables(file)
        except Exception as e:
            # load_landing()
            st.error(f"Error loading tables: {e}")
            st.stop()
            
        selected_table = main_container.selectbox("Select a table:", tables)
        file.seek(0)
        try:
            df = load_sqlite_data(file_type, file=file, sqlite_table=selected_table)
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()
    else:
        has_headers = main_container.selectbox("Does the file have column headers?", [True, False], index=0)
        try:
            df = load_file_data(file, file_type, has_headers, excel_sheet)
        except Exception as e:
            # load_landing()
            st.error(f"Error loading data: {e}")
            st.stop()

    if st.button("Connect"):
        st.session_state.df = df
        st.session_state.active_tab = 0
        st.rerun()

def server_upload(main_container):
    db_type = main_container.selectbox("Database type:", ["PostgreSQL", "MySQL", "SQL Server"])
    user = main_container.text_input("Username")
    password = main_container.text_input("Password", type="password")
    host = main_container.text_input("Host", value="localhost")
    port = main_container.text_input("Port", value="5432")
    database = main_container.text_input("Database name")

    try:
        tables = get_db_tables(db_type.lower().replace(" ", ""), user, password, host, port, database)
        selected_table = main_container.selectbox("Select a table:", tables)
        df = load_db_data(db_type.lower().replace(" ", ""), user, password, host, port, database, table_name=selected_table)
    except Exception as e:
        # load_landing()
        st.error(f"Error loading tables or data: {e}")
        st.stop()

    if st.button("Connect"):
        st.session_state.df = df
        st.session_state.active_tab = 0
        st.rerun()

# Initialize session state for tab navigation
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = -1

# Define tab names
tab_names = ["Data Preview", "Topic Extraction", "Feature Extraction", "Recommendations", ""]

# Function to navigate to next tab
def next_tab(max_tabs):
    if st.session_state.active_tab < max_tabs - 2:
        st.session_state.active_tab += 1
        # st.rerun()

def prev_tab():
    if st.session_state.active_tab >= 0:
        st.session_state.active_tab -= 1

def disconnect():
    st.session_state.active_tab = -1

# Display tab navigation buttons
col1, col2, col3, col4 = st.columns([2, 2, 4, 2], vertical_alignment='center')
with col1:
    if st.session_state.active_tab >= 0:
        st.button("⬅️ Previous", on_click=prev_tab)
with col2:
    if st.session_state.active_tab != -1:
        st.button("Disconnect", on_click=disconnect)
with col3:
    st.markdown(f"**{tab_names[st.session_state.active_tab]}**", unsafe_allow_html=True)
with col4:
    if st.session_state.active_tab != -1 and st.session_state.active_tab < len(tab_names) - 2:
        st.button("➡️ Next", on_click=next_tab, args=[len(tab_names)])

st.write("debug" + str(st.session_state.active_tab))
main_container = st.container()

if st.session_state.active_tab == -1:
    st.session_state.df = None
    if source_type == "File":
        file_upload(main_container)
    else:
        server_upload(main_container)

if st.session_state.df is None:
    st.stop()

df = st.session_state.df
st.sidebar.header("Column Selection")
selected_column = st.sidebar.selectbox("Select a text column:",
                                       [None] + list(df.columns))
remaining_columns = set(df.columns).difference([selected_column])
entity_column = st.sidebar.selectbox("Select a column for key entity:",
                                     [None] + list(remaining_columns))

# Display tab content
with main_container:
    # instantiate all required dataframes
    if "final_topics_df" not in st.session_state:
        st.session_state.final_topics_df = None
    if "associations_df" not in st.session_state:
        st.session_state.associations_df = None

    if st.session_state.active_tab == 0:  # Data Preview
        st.subheader("Preview of Your Data")
        if df is None or df.empty:
            st.warning("This table doesn't seem to have data. Please check your data source again.")
        else:
            st.dataframe(df, use_container_width=True)

    elif st.session_state.active_tab == 1:  # Topic Extraction
        st.subheader("Generated Topics")
        new_topics_df = None
        
        if df is None or df.empty or not selected_column:
            st.info("Please upload data and select a text column to view topics.")
        else:
            topics_df = extract_topics(df=df, selected_column=selected_column)
            st.write("These are the topics extracted from the provided column. "
                     "Feel free to make changes and press Approve when you feel the topics "
                     "accurately describe your data.")
            with st.form("Topics Reduction"):
                new_topics_df = st.data_editor(topics_df, num_rows="dynamic", use_container_width=True)
                is_approved = st.form_submit_button("Approve Changes")

        if new_topics_df is None or new_topics_df.empty:
            pass
        elif not is_approved and st.session_state.final_topics_df is None:
            st.info("Please press 'Approve Changes' to continue.")
        else:
            st.subheader("Final Topics")
            final_topics_df = reduce_topics(df=new_topics_df)
            st.write("These are the topics that you have approved. Further analysis will be based on these.")
            st.dataframe(final_topics_df, use_container_width=True)
            st.session_state.final_topics_df = final_topics_df

    elif st.session_state.active_tab == 2:  # Feature Extraction
        st.subheader("Extracted Features")
        final_topics_df = st.session_state.final_topics_df
        if final_topics_df is None or final_topics_df.empty:
            st.info("Please approve topics and select an entity column to view extracted features.")
        else:
            associations_df = extract_features(
                df=df,
                topics_df=final_topics_df,
                selected_column=selected_column,
                entity_column=entity_column
            )
            st.write("These are the insights and topic associations derived from the data.")
            st.dataframe(associations_df, use_container_width=True)
            logger.info("Associations with NEUTRAL sentiment: %d",
                        associations_df[associations_df['sentiment'] == "NEUTRAL"].shape[0])
            st.session_state.associations_df = associations_df

    elif st.session_state.active_tab == 3:  # Recommendations
        st.subheader("Recommendations")
        associations_df = st.session_state.associations_df
        if associations_df is None or associations_df.empty:
            st.info("Please complete feature extraction to view recommendations.")
        else:
            topic_column = entity_column if entity_column is not None else "topic"
            grouped_df = create_recommendation_table(associations_df, topic_column)

            st.subheader("Topic Breakdown:")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Topic Distribution by Document Count")
                topic_count_df = create_topic_count_table(associations_df, topic_column)
                
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=topic_count_df[['POSITIVE', 'NEGATIVE', 'NEUTRAL']].values,
                    x=['POSITIVE', 'NEGATIVE', 'NEUTRAL'],
                    y=topic_count_df[topic_column],
                    colorscale='YlGnBu',
                    text=topic_count_df[['POSITIVE', 'NEGATIVE', 'NEUTRAL']].values,
                    texttemplate="%{text}",
                    textfont={"size": 12}
                ))
                fig_heatmap.update_layout(
                    xaxis_title="Sentiment",
                    yaxis_title="Topics" if topic_column == 'topic' else "Entities",
                    height=500,
                    title="Heatmap of Topic Distribution"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)

            with col2:
                st.write("Topics with Highest Strength (Negative Sentiment)")
                grouped_df = create_recommendation_table(associations_df, topic_column)
                top_strength_df = grouped_df.head(5)
                
                fig_bar = px.bar(
                    top_strength_df,
                    x=top_strength_df.index,
                    y='strength',
                    text=top_strength_df['strength'].round(2),
                    color='strength',
                    color_continuous_scale='Reds',
                    title='Top Topics by Negative Sentiment Strength'
                )
                fig_bar.update_traces(textposition='auto')
                fig_bar.update_layout(
                    xaxis_title="Topics" if topic_column == 'topic' else "Entities",
                    yaxis_title="Average Strength",
                    height=500,
                    showlegend=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            recommend_df = grouped_df.iloc[:min(grouped_df.shape[0], 5)]
            st.dataframe(recommend_df)
            topic_issues_pairs = list(zip(
                recommend_df.index,
                recommend_df['strength'],
                recommend_df['issues']
            ))
            conclusion = create_conclusion(topic_issues_pairs)
            st.markdown("## Conclusion")
            st.markdown(conclusion)