import os
filepath = r"C:\Vs code Automation\Trend Alpha 4.0\dashboard.py"

with open(filepath, 'r', encoding='utf-8') as f:
    c = f.read()

# ── FIX 3: Replace redundant get_global_mf_name_map() with cached version ──
old3 = '            _global_mf_n = get_global_mf_name_map()\n'
new3 = '            _global_mf_n = _mf_name_map_global  # reuse cached from line 4882\n'
c = c.replace(old3, new3, 1)  # replace only first occurrence

# ── FIX 4: Wrap category analysis in try/except ──
old4 = '            st.markdown("##### \U0001f4c2 Category-Based Analysis")\n\n            if "Category" in df_core_t.columns:'
new4 = '            st.markdown("##### \U0001f4c2 Category-Based Analysis")\n\n            try:\n                if "Category" in df_core_t.columns:'
c = c.replace(old4, new4, 1)

# Add closing except
old4b = '            st.markdown("".join(_core_html_t), unsafe_allow_html=True)\n    else:\n        st.info("Core Portfolio Universe file not found for this date. Run the pipeline to generate L1_Core_Universe.csv.")'
new4b = '            st.markdown("".join(_core_html_t), unsafe_allow_html=True)\n            except Exception as _e_cat:\n                st.warning(f"\u26a0\ufe0f Category analysis table could not be rendered: {_e_cat}")\n    else:\n        st.info("Core Portfolio Universe file not found for this date. Run the pipeline to generate L1_Core_Universe.csv.")'
c = c.replace(old4b, new4b, 1)

# ── FIX 5: Named constants for selection weights ──
old5 = '                        # Composite Momentum Score\n                        _comp_score = (0.20 * _r15d) + (0.30 * _r1m) + (0.25 * _r2m) + (0.15 * _r3m) + (0.10 * _r6m)'
new5 = '                        # Composite Momentum Score (time-decay weighted)\n                        _W_15D = 0.20; _W_1M = 0.30; _W_2M = 0.25; _W_3M = 0.15; _W_6M = 0.10\n                        _comp_score = (_W_15D * _r15d) + (_W_1M * _r1m) + (_W_2M * _r2m) + (_W_3M * _r3m) + (_W_6M * _r6m)'
c = c.replace(old5, new5, 1)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(c)

print('Fixes 3, 4, 5 applied OK')
