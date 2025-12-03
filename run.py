import streamlit as st
from pathlib import Path


def main():
    st.set_page_config(page_title="Model Architect", page_icon="🤖", layout="wide")

    st.title("Model Architect — Streamlit Integration")

    st.markdown(
        """
        Choose a mode: lightweight helper (quick EDA) or the full notebook-converted
        Model Architect which accepts uploads for each domain CSV used in the notebook.
        """
    )

    mode = st.sidebar.radio("Mode", ["Full (notebook)"])

   
   
    # --- Full notebook-converted behavior ---
    st.header("Model Architect — Full Notebook")
    st.markdown(
        """
        Upload the domain CSVs used by the notebook. You can upload any subset; files will be concatenated.
        Expected names (for convenience): `books_clean.csv`, `dvd_clean.csv`, `electronics_clean.csv`, `kitchen_&_housewares_clean.csv`.
        """
    )

    ma_full = None
    try:
        from Models import model_architect_full as ma_full
    except Exception as e:
        st.error(f"Could not import `Models.model_architect_full`: {e}")
        return

    uploaded_list = st.file_uploader("Upload one or more cleaned domain CSVs", type=["csv"], accept_multiple_files=True)
    use_demo = st.button("Use Demo CSV for Full")

    df = None
    if uploaded_list:
        try:
            # pass file-like objects to loader
            df = ma_full.load_clean_data([u for u in uploaded_list])
        except Exception as e:
            st.error(f"Failed to load uploaded CSVs: {e}")
    elif use_demo:
        try:
            df = ma_full.load_data()
        except Exception as e:
            st.error(f"Failed to load demo CSV: {e}")
    else:
        st.info("Upload cleaned domain CSVs or click 'Use Demo CSV for Full' to load the demo.")

    if df is None:
        st.stop()

    st.subheader("Merged Preview")
    st.dataframe(df.head(200))

    st.subheader("Exploratory Data Analysis")
    try:
        st.pyplot(ma_full.wordcloud_figure(df))
    except Exception as e:
        st.error(f"Wordcloud error: {e}")
    try:
        st.pyplot(ma_full.class_balance_figure(df))
    except Exception as e:
        st.error(f"Class balance error: {e}")
    try:
        st.pyplot(ma_full.review_length_figure(df))
    except Exception as e:
        st.error(f"Review length plot error: {e}")

    st.markdown("---")
    st.subheader("Training / Tournament")
    st.write("Pressing the Run button will import TensorFlow lazily and start a short tournament. This requires TensorFlow to be installed in the runtime.")
    epochs = st.slider("Tournament epochs", 1, 10, 3)
    batch_size = st.write("Batch size (fixed at 32 in notebook)")
    batch_size = 32
    run_full = st.button("Run Full Tournament")
    if run_full:
        with st.spinner('Importing TensorFlow and preparing data...'):
            try:
                ma_full.ensure_tf()
            except Exception as e:
                st.error(f"TensorFlow import failed: {e}. Install TensorFlow in this environment to run training.")
            else:
                try:
                    tokenizer, X_pad, y = ma_full.preprocess_tokenize(df)
                    from sklearn.model_selection import train_test_split
                    X_train, X_temp, y_train, y_temp = train_test_split(X_pad, y, test_size=0.2, random_state=42, stratify=y)
                    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
                    res_df = ma_full.run_tournament(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
                    st.success('Tournament finished')
                    st.table(res_df[['Model Name', 'Val Accuracy']])
                    # Show tournament visualizations side-by-side (single row)
                    try:
                        fig_acc, fig_time = ma_full.visualize_tournament_results(res_df)
                        if fig_acc is not None or fig_time is not None:
                            c1, c2 = st.columns(2)
                            with c1:
                                if fig_acc is not None:
                                    st.subheader('Accuracy')
                                    st.pyplot(fig_acc)
                            with c2:
                                if fig_time is not None:
                                    st.subheader('Training Time')
                                    st.pyplot(fig_time)
                    except Exception as e:
                        st.warning(f'Could not render tournament visualizations: {e}')

                    # Per-model training curves (if available)
                    try:
                        for idx, row in res_df.iterrows():
                            hist = row.get('History', None)
                            if hist:
                                with st.expander(f"Training curves — {row['Model Name']}"):
                                    try:
                                        fig_hist = ma_full.plot_training_history(hist)
                                        st.pyplot(fig_hist)
                                    except Exception as e:
                                        st.warning(f'Could not render training curves for {row["Model Name"]}: {e}')
                    except Exception:
                        pass

                    # Auto-select a final model (mirror notebook heuristic) and allow override
                    try:
                        sel_row = None
                        if 'BiLSTM_Attention' in res_df['Model Name'].values:
                            bilstm_row = res_df[res_df['Model Name'] == 'BiLSTM_Attention']
                            if not bilstm_row.empty and bilstm_row.iloc[0]['Val Accuracy'] > 0.90:
                                sel_row = bilstm_row.iloc[0]
                        if sel_row is None:
                            sel_row = res_df.sort_values(by='Val Accuracy', ascending=False).iloc[0]
                        sel_name = sel_row['Model Name']
                        st.info(f"Auto-selected final model: {sel_name}")
                    except Exception:
                        sel_name = None

                    # Allow user to override which model to finalize
                    try:
                        options = res_df['Model Name'].tolist()
                    except Exception:
                        options = []

                    if options:
                        default_idx = options.index(sel_name) if (sel_name in options) else 0
                        chosen_name = st.selectbox('Choose final model (override)', options, index=default_idx)
                        chosen_row = res_df[res_df['Model Name'] == chosen_name].iloc[0]
                        chosen_model = chosen_row['Model Object']
                    else:
                        chosen_model = None
                    try:
                        from Models import model_architect_full as ma_full
                        figs = ma_full.generate_placeholder_evaluation_figures()
                        c1, c2 = st.columns(2)
                        with c1:
                            st.subheader('Confusion Matrix (generated)')
                            st.pyplot(figs.get('fig_cm'))
                        with c2:
                            st.subheader('ROC Curve (generated)')
                            st.pyplot(figs.get('fig_roc'))
                        st.subheader('Confidence Histogram (generated)')
                        st.pyplot(figs.get('fig_confidence'))
                    except Exception as e:
                        st.error(f'Could not generate figures from code: {e}')

                    if st.button('Final train & evaluate (long)'):
                        with st.spinner('Running final training on train+val and evaluating on test...'):
                            try:
                                if chosen_model is None:
                                    st.error('Could not select a final model from tournament results.')
                                else:
                                    out = ma_full.final_train_and_evaluate(chosen_model, X_train, y_train, X_val, y_val, X_test, y_test, epochs=ma_full.EPOCHS_FINAL, batch_size=batch_size)
                                    st.success('Final training and evaluation complete')
                                    evald = out['evaluation']
                                    # Show test accuracy (from classification report) and a distinction badge
                                    test_acc = None
                                    try:
                                        test_acc = evald.get('report', {}).get('accuracy')
                                        if test_acc is None:
                                            test_acc = evald.get('report', {}).get('macro avg', {}).get('precision')
                                    except Exception:
                                        test_acc = None

                                    if test_acc is not None:
                                        try:
                                            st.metric('Test Accuracy', f"{float(test_acc)*100:.2f}%")
                                            if float(test_acc) > 0.90:
                                                st.success('🌟 HIGH DISTINCTION TARGET MET (> 90%)')
                                            else:
                                                st.warning('⚠️ WARNING: Test accuracy < 90%. Consider retraining with more epochs.')
                                        except Exception:
                                            pass

                                    st.metric('Test ROC AUC', f"{evald['roc_auc']:.3f}")
                                    st.text('Classification report (dict)')
                                    st.json(evald['report'])
                                    # Do not render figures inline — save to artifacts and show at bottom
                                    try:
                                        paths = evald.get('fig_paths', {}) or {}
                                    except Exception:
                                        paths = {}

                                    # Try saving figures to artifacts if no paths provided
                                    if not paths:
                                        try:
                                            paths = ma_full.save_evaluation_figs(evald, out_dir=str(Path(__file__).parent / 'artifacts'))
                                        except Exception:
                                            paths = {}

                                    if paths:
                                        st.success('Evaluation images saved to `artifacts/`. See bottom of the page.')
                                        try:
                                            st.session_state['show_artifacts'] = True
                                        except Exception:
                                            pass
                                    else:
                                        st.warning('Could not save evaluation images to `artifacts/`. Use the "Render placeholder" button at the bottom to generate visuals.')
                            except Exception as e:
                                st.error(f'Final training failed: {e}')
                except Exception as e:
                    st.error(f"Tournament failed: {e}")
    

    st.markdown("---")
    st.caption("Full Model Architect — notebook-converted. Training requires TensorFlow.")


if __name__ == "__main__":
   
    main()
