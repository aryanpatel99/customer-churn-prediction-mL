def render_custom_table(df):
    # Format dollars and percentages
    display_df = df.copy()
    display_df["Monthly Charge"] = display_df["Monthly Charge"].apply(lambda x: f"${x:.2f}")
    
    html = display_df.to_html(index=False, classes="custom-table", escape=False)
    wrap = f"""
    <div class="table-container">
        {html}
    </div>
    """
    return wrap
