# streamlit_app.py
import streamlit as st
import os
from datetime import datetime

# Snowflake Snowpark session (works inside Streamlit-in-Snowflake)
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import Session

# Import your unified agent pipeline
from agents import evaluate_ticket
from Agents.classifier_agent import ticket_classification
from Agents.sentiment_agent import classify_sentiment
from Agents.escalation_agent import escalation

# --------------------------------------------
# SESSION HANDLING
# --------------------------------------------
def get_session() -> Session:
    """Try to get Snowflake session (inside app or local)."""
    try:
        return get_active_session()
    except Exception:
        from Integrations.snowflake_conn import connect_snowflake
        conn = connect_snowflake()
        return Session.builder.configs({}).connection(conn).create()


# --------------------------------------------
# SAVE TO SNOWFLAKE
# --------------------------------------------
def log_to_snowflake(session: Session, user_message: str, bot_response: str,
                     category: str, sentiment: str, escalation_decision: str):
    """Insert message pair + metadata into SUPPORT_TICKETS and TICKET_RESULTS tables."""
    try:
        # Insert ticket text
        session.sql(
            "INSERT INTO SUPPORT_TICKETS (USER_TEXT) VALUES (%s)",
            [user_message]
        ).collect()

        # Insert result metadata
        session.sql(
            """
            INSERT INTO TICKET_RESULTS
              (TICKET_ID, PREDICTED_CATEGORY, SENTIMENT, SENTIMENT_CONFIDENCE, ESCALATION_DECISION, REASON)
            SELECT MAX(TICKET_ID), %s, %s, %s, %s, %s FROM SUPPORT_TICKETS
            """,
            [category, sentiment, 1.0, escalation_decision, "Generated via Streamlit App"]
        ).collect()

        # Optional: add full conversation record table
        session.sql(
            """
            CREATE TABLE IF NOT EXISTS CHAT_LOGS (
                CHAT_ID INTEGER AUTOINCREMENT,
                USER_MESSAGE STRING,
                BOT_RESPONSE STRING,
                CATEGORY STRING,
                SENTIMENT STRING,
                ESCALATION STRING,
                CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        ).collect()

        session.sql(
            """
            INSERT INTO CHAT_LOGS (USER_MESSAGE, BOT_RESPONSE, CATEGORY, SENTIMENT, ESCALATION)
            VALUES (%s, %s, %s, %s, %s)
            """,
            [user_message, bot_response, category, sentiment, escalation_decision]
        ).collect()
    except Exception as e:
        st.warning(f"⚠️ Could not log to Snowflake: {e}")


# --------------------------------------------
# STREAMLIT APP
# --------------------------------------------
def main():
    st.set_page_config(page_title="Support AI Chatbot", page_icon="🤖", layout="centered")
    st.title("🤖 Support AI Chatbot (Snowflake Edition)")
    st.caption("Runs classification → sentiment → escalation → GPT-3.5 response pipeline")

    session = get_session()
    try:
        info = session.sql("SELECT CURRENT_USER(), CURRENT_DATABASE(), CURRENT_SCHEMA()").collect()
        st.success(f"Connected as: `{info[0][0]}` | DB: `{info[0][1]}` | Schema: `{info[0][2]}`")
    except Exception:
        st.warning("⚠️ Running locally (not connected to active Snowflake session)")

    # Maintain chat memory
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User input
    user_message = st.text_area("💬 Enter your support query:", height=150)

    if st.button("Send"):
        if not user_message.strip():
            st.warning("Please type a message first.")
            return

        with st.spinner("Analyzing and generating response..."):
            # --- Run full evaluation ---
            classification = ticket_classification(user_message)
            sentiment = classify_sentiment(user_message)
            escalation_decision = escalation(sentiment)
            final_response = evaluate_ticket(user_message)

            # --- Save to chat history ---
            st.session_state.chat_history.append({
                "user": user_message,
                "bot": final_response,
                "classification": classification,
                "sentiment": sentiment,
                "escalation": escalation_decision
            })

            # --- Log to Snowflake ---
            log_to_snowflake(session, user_message, final_response,
                             classification, str(sentiment), escalation_decision)

        st.success("Response generated and logged successfully ✅")

    # Display chat
    for msg in reversed(st.session_state.chat_history):
        st.markdown(f"**🧍 User:** {msg['user']}**")
        st.markdown(f"**🤖 Bot:** {msg['bot']}**")
        st.caption(
            f"_Category: {msg['classification']} | Sentiment: {msg['sentiment']} | Escalation: {msg['escalation']}_"
        )
        st.divider()

    # Clear chat
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.experimental_rerun()


if __name__ == "__main__":
    main()
