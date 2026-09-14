import streamlit as st
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

# --- 1. CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Smart CRM B2B", page_icon="📈", layout="wide")

# --- 2. GESTIONE MULTI-TENANT ---

@st.cache_data
def load_data_cyber():
    df_crm = pd.read_csv("frontend_data/dataset_crm.csv")
    df_ml = pd.read_csv("frontend_data/df_ml_ready.csv", index_col=0)
    return df_crm, df_ml

@st.cache_data
def load_data_horeca():
    # Leggiamo il csv.
    df_crm = pd.read_csv("frontend_data/dataset_horeca.csv")
    df_ml = pd.read_csv("frontend_data/df_ml_ready_horeca.csv", index_col=0)
    
    # ADAPTER PATTERN: Traduciamo le colonne del dataset
    mappatura_colonne = {
        'local_name_unique': 'Ragione_Sociale',
        'macro_area': 'Macro_Area',
        'avg_spend_eur_synth': 'Scontrino_Medio'
    }
    df_crm = df_crm.rename(columns=mappatura_colonne)
    return df_crm, df_ml

@st.cache_resource
def train_models(df_machine_learning, prodotti):
    # Escludiamo sia i prodotti Cyber (Prod_) sia i target HoReCa (y_)
    features_anagrafiche = [
        col for col in df_machine_learning.columns 
        if not col.startswith('Prod_') and not col.startswith('y_')
    ]
    X_train = df_machine_learning[features_anagrafiche]
    
    modelli_lda = {}
    modelli_lr = {}
    
    for prodotto in prodotti:
        y_train = df_machine_learning[prodotto]
        
        # Addestra LDA
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)
        modelli_lda[prodotto] = lda
        
        # Addestra Logistic Regression
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        modelli_lr[prodotto] = lr
        
    return modelli_lda, modelli_lr, features_anagrafiche

# --- 3. MENU LATERALE E CONTEXT SWITCHING ---
st.sidebar.title("🏢 Seleziona Tenant")
dominio = st.sidebar.radio(
    "Ambiente di lavoro:",
    ["B2B Cybersecurity", "Ho.Re.Ca."],
    key="tenant_switch"
)

st.sidebar.divider()
st.sidebar.title("Navigazione CRM")
menu = st.sidebar.radio(
    "Vai a:",
    ["🏠 Dashboard Intelligente", "🏢 Anagrafica", "👥 Contatti", "🤝 Offerte"],
    key="main_menu_radio"
)

# --- BOTTONE DI RETRAINING ON-DEMAND ---
st.sidebar.divider()
st.sidebar.markdown("### 🛠️ Area Admin")
if st.sidebar.button("🔄 Forza riaddestramento modelli"):
    # 1. Svuota la cache dei dataframe (CSV)
    st.cache_data.clear()
    # 2. Svuota la cache dei modelli addestrati (LDA e LR)
    st.cache_resource.clear()
    
    st.sidebar.success("Cache svuotata! Ricalcolo dei pesi in corso...")
    # 3. Riavvia l'app per forzare l'esecuzione da zero
    st.rerun()

# --- INIZIALIZZAZIONE DEL CONTESTO ---
if dominio == "B2B Cybersecurity":
    df_crm, df_ml = load_data_cyber()
    prodotti_db = ['Prod_Firewall', 'Prod_Endpoint_Security', 'Prod_Cloud_Security', 'Prod_SOC_Gestito']
    modelli_lda, modelli_lr, features = train_models(df_ml, prodotti_db)
    
    opzioni_prodotti = {
        'Prod_Firewall': '🛡️ Firewall Base',
        'Prod_Endpoint_Security': '💻 Endpoint Security',
        'Prod_Cloud_Security': '☁️ Cloud Security',
        'Prod_SOC_Gestito': '🚨 SOC Gestito (Premium)'
    }
    default_soglia = {
        'Prod_Firewall': 50, 'Prod_Endpoint_Security': 40, 
        'Prod_Cloud_Security': 35, 'Prod_SOC_Gestito': 20
    }
    label_metrica_fatturato = "Fatturato (€)"

elif dominio == "Ho.Re.Ca.":
    df_crm, df_ml = load_data_horeca()
    
    # 1. Usiamo i nomi delle colonne target del dataset reale
    prodotti_db = ['y_manager', 'y_pay', 'y_giftcard', 'y_pro']
    modelli_lda, modelli_lr, features = train_models(df_ml, prodotti_db)
    
    # 2. Mappiamo i nomi per la UI del selettore
    opzioni_prodotti = {
        'y_manager': '🍽️ Booking Manager (Gestione Prenotazioni)',
        'y_pay': '💳 Smart Pay (Integrazione Fintech)',
        'y_giftcard': '🎁 Circuito Gift Card (Welfare Aziendale)',
        'y_pro': '📈 Yield Manager PRO (Ottimizzazione Ricavi)'
    }
    
    # 3. Adattiamo le soglie di default ai nuovi target
    default_soglia = {
        'y_manager': 40, 'y_pay': 35, 
        'y_giftcard': 30, 'y_pro': 25
    }
    label_metrica_fatturato = "Scontrino Medio (€)"

# --- 4. GESTIONE DELLE PAGINE ---

if menu == "🏠 Dashboard Intelligente":
    st.title(f"🎯 Dashboard Intelligente ({dominio})")
    
    # ==========================================================
    # FUNZIONI DI SUPPORTO
    # ==========================================================
    def pulisci_nome_variabile(nome_var):
        traduzioni = {
            'macro_area_tourist_presences_2024_m_istat': 'Presenze Turistiche ISTAT',
            'avg_spend_eur_synth': 'Scontrino Medio Stimato',
            'seats_synth': 'Coperti Stimati',
            'employees_synth': 'Forza Lavoro (Dipendenti)',
            'revenue_eur_synth': 'Fatturato Stimato',
            'profit_margin_derived_synth': 'Margine di Profitto',
            'digital_presence_score_synth': 'Presenza Digitale (Score)',
            'restaurants_1km_synth': 'Competitor Entro 1km',
            'reputation_score_synth': 'Reputazione Web',
            'restaurant_type_': 'Categoria: ',
            'macro_area_': 'Area: '
        }
        nome_pulito = nome_var
        for eng, ita in traduzioni.items():
            if eng in nome_pulito:
                nome_pulito = nome_pulito.replace(eng, ita)
        return nome_pulito.replace('_', ' ').title()

    # --- FILTRI DI BUSINESS E MODELLO ---
    st.markdown("### ⚙️ Impostazioni Campagna")
    col_filtro1, col_filtro2, col_filtro3, col_filtro4 = st.columns(4)
    
    with col_filtro1:
        target_selezionato = st.selectbox(
            "Prodotto/servizio:", 
            list(opzioni_prodotti.keys()), 
            format_func=lambda x: opzioni_prodotti[x]
        )
        
    with col_filtro2:
        modello_scelto = st.selectbox(
            "Algoritmo ML:", 
            ["Logistic Regression (LR)", "Linear Discriminant Analysis (LDA)"]
        )

    with col_filtro3:
        lista_aree = ["Tutta Italia"] + list(df_crm['Macro_Area'].dropna().unique())
        area_scelta = st.selectbox("Area Geografica:", lista_aree)
        
    with col_filtro4:
        soglia_visiva = st.slider(
            "Soglia minima (Hot Lead %):", 
            min_value=5, max_value=90, value=default_soglia[target_selezionato], step=5
        )
        soglia = soglia_visiva / 100.0 
        
    st.divider()

    # --- CALCOLO PROPENSITY SCORE REAL-TIME ---
    maschera_prospect = df_ml[target_selezionato] == 0
    df_prospect_ml = df_ml.loc[maschera_prospect, features]
    df_prospect_ui = df_crm.loc[maschera_prospect.values].copy()
    
    if area_scelta != "Tutta Italia":
        maschera_area = df_prospect_ui['Macro_Area'] == area_scelta
        df_prospect_ui = df_prospect_ui[maschera_area]
        df_prospect_ml = df_prospect_ml.loc[maschera_area.values]
    
    if "LDA" in modello_scelto:
        modello_attivo = modelli_lda[target_selezionato]
    else:
        modello_attivo = modelli_lr[target_selezionato]
        
    if len(df_prospect_ml) > 0:
        probabilita = modello_attivo.predict_proba(df_prospect_ml)[:, 1]
    else:
        probabilita = np.array([])
    
    df_prospect_ui['Propensity_Score'] = probabilita
    
    hot_leads = df_prospect_ui[df_prospect_ui['Propensity_Score'] >= soglia].sort_values(by='Propensity_Score', ascending=False)
    
    # --- KPI E METRICHE CON INTERVALLO DI CONFIDENZA ---
    st.markdown("### 📊 Overview Bacino Prospect")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Totale Prospect in Target", len(df_prospect_ui))
    kpi2.metric(f"Hot Leads (>{soglia_visiva}%)", len(hot_leads))
    
    if len(hot_leads) > 0:
        tasso_conv_stimato = hot_leads['Propensity_Score'].mean()
        varianza = np.sum(hot_leads['Propensity_Score'] * (1 - hot_leads['Propensity_Score']))
        standard_error = np.sqrt(varianza) / len(hot_leads)
        margine_errore = 1.96 * standard_error * 100
        kpi3.metric("Tasso di Conversione Stimato", f"{tasso_conv_stimato * 100:.1f}%", f"± {margine_errore:.1f}% (Conf. 95%)", delta_color="off")
    else:
        kpi3.metric("Tasso di Conversione Stimato", "0.0%")
    
    st.divider()
    
    # --- TABELLA E EXPLAINABILITY GLOBALE ---
    col_tabella, col_xai = st.columns([2, 1])
    
    with col_tabella:
        st.markdown("### 🏆 Top Prospect da contattare")
        
        col_finanziaria = 'Fatturato_Sintetico' if dominio == "B2B Cybersecurity" else 'Scontrino_Medio'
        colonne_da_mostrare = ['Ragione_Sociale', 'Macro_Area', col_finanziaria, 'Propensity_Score']
        
        st.dataframe(
            hot_leads[colonne_da_mostrare].head(15), 
            width="stretch",
            column_config={
                col_finanziaria: st.column_config.NumberColumn(label_metrica_fatturato, format="%d"),
                "Propensity_Score": st.column_config.ProgressColumn("Propensione (%)", min_value=0.0, max_value=1.0, format="%.2f")
            }
        )

        # SELETTORE CLIENTE
        st.markdown("---")
        cliente_selezionato = st.selectbox(
            "🔍 Seleziona un'azienda dalla lista per generare la scheda commerciale:", 
            options=["Seleziona un'azienda..."] + list(hot_leads['Ragione_Sociale'].values)
        )

    with col_xai:
        st.markdown("### 🧠 Explainability Globale")
        nome_modello_ui = "della LDA" if "LDA" in modello_scelto else "del Regressore Logistico"
        st.info(f"Cosa cerca l'algoritmo **{nome_modello_ui}** per suggerire {opzioni_prodotti[target_selezionato]}?")
        
        pesi = modello_attivo.coef_[0]
        df_pesi = pd.DataFrame({'Variabile': features, 'Peso': pesi})
        
        driver_positivi = df_pesi[df_pesi['Peso'] > 0].sort_values(by='Peso', ascending=False).head(3)
        driver_negativi = df_pesi[df_pesi['Peso'] < 0].sort_values(by='Peso', ascending=True).head(3)

        st.write("**🟢 Driver Positivi (Spingono all'acquisto):**")
        if not driver_positivi.empty:
            for _, row in driver_positivi.iterrows():
                st.markdown(f"* **{pulisci_nome_variabile(row['Variabile'])}** (+{row['Peso']:.2f})")
        else:
            st.markdown("*Nessun driver positivo forte.*")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.write("**🔴 Fattori Penalizzanti (Zavorrano lo score):**")
        if not driver_negativi.empty:
            for _, row in driver_negativi.iterrows():
                st.markdown(f"* **{pulisci_nome_variabile(row['Variabile'])}** ({row['Peso']:.2f})")
        else:
            st.markdown("*Nessun fattore penalizzante forte.*")
        
    # --- DETTAGLIO CLIENTE E EXPLAINABILITY LOCALE ---
    if cliente_selezionato != "Seleziona un'azienda...":
        st.divider()
        
        # 1. Trova l'indice numerico del cliente
        idx_cliente = hot_leads[hot_leads['Ragione_Sociale'] == cliente_selezionato].index[0]
        
        dati_cliente_crm = hot_leads.loc[idx_cliente]
        # FIX DEFINITIVO: Usiamo .iloc per pescare tramite la posizione numerica (es. 119)
        dati_cliente_ml = df_ml.iloc[idx_cliente][features]
        
        # 2. Calcolo dei Pesi Locali
        pesi_globali = modello_attivo.coef_[0]
        impatto_locale = dati_cliente_ml.values * pesi_globali
        
        df_impatto = pd.DataFrame({
            'Variabile': features,
            'Impatto': impatto_locale,
            'Valore_Modello': dati_cliente_ml.values 
        })
        
        driver_loc_pos = df_impatto[df_impatto['Impatto'] > 0].sort_values(by='Impatto', ascending=False).head(3)
        driver_loc_neg = df_impatto[df_impatto['Impatto'] < 0].sort_values(by='Impatto', ascending=True).head(3)        
        # 3. RENDERIZZAZIONE SCHEDA NELL'EXPANDER
        with st.expander(f"👤 Apri Scheda Commerciale per: **{cliente_selezionato}**", expanded=True):
            
            # Riga 1: Metriche CRM principali
            col_crm1, col_crm2, col_crm3, col_crm4 = st.columns(4)
            col_crm1.metric("Propensity Score", f"{dati_cliente_crm['Propensity_Score']*100:.1f}%")
            col_crm2.metric("Macro Area", dati_cliente_crm['Macro_Area'])
            
            if dominio == "Ho.Re.Ca.":
                col_crm3.metric("Scontrino Medio", f"€ {dati_cliente_crm.get('Scontrino_Medio', 0):.2f}")
                col_crm4.metric("Dipendenti/Coperti", dati_cliente_crm.get('employees_synth', 'N/D'))
            else:
                col_crm3.metric("Fatturato", f"€ {dati_cliente_crm.get('Fatturato_Sintetico', 0):.2f}")
                col_crm4.metric("Dipendenti", dati_cliente_crm.get('Dipendenti', 'N/D'))
                
            st.markdown("---")
            
            # Riga 2: Explainability Locale
            st.markdown(f"##### 🧠 Leve negoziali per vendere **{opzioni_prodotti[target_selezionato]}** a questo cliente:")
            col_xai_pos, col_xai_neg = st.columns(2)
            
            with col_xai_pos:
                st.success("**✅ Argomenti a favore (Driver Locali)**")
                for _, row in driver_loc_pos.iterrows():
                    impatto = row['Impatto']
                    valore = row['Valore_Modello']
                    # Mostriamo l'impatto con il segno + davanti
                    st.write(f"- **{pulisci_nome_variabile(row['Variabile'])}** (+{impatto:.2f})")
                    
            with col_xai_neg:
                st.warning("**⚠️ Ostacoli da superare (Zavorre Locali)**")
                for _, row in driver_loc_neg.iterrows():
                    impatto = row['Impatto']
                    valore = row['Valore_Modello']
                    # L'impatto è già negativo, quindi il segno - appare da solo
                    st.write(f"- **{pulisci_nome_variabile(row['Variabile'])}** ({impatto:.2f})")

elif menu == "🏢 Anagrafica":
    st.title(f"🏢 Modulo Anagrafica ({dominio})")
    st.info("Censimento dei clienti e dei potenziali clienti.")
    
    with st.form("nuova_anagrafica"):
        st.subheader("Inserisci Nuova Scheda")
        col1, col2 = st.columns(2)
        with col1:
            ragione_sociale = st.text_input("Ragione Sociale / Nome Locale")
            partita_iva = st.text_input("Partita IVA")
            tipo = st.selectbox("Tipo", ["Cliente", "Prospect", "Partner"])
        with col2:
            regione = st.selectbox("Macro Area", ["Nord", "Centro", "Sud_Isole"])
            if dominio == "B2B Cybersecurity":
                metrica_1 = st.number_input("Fatturato Previsto (€)", min_value=0)
                metrica_2 = st.number_input("Dipendenti", min_value=1)
            else:
                metrica_1 = st.number_input("Scontrino Medio (€)", min_value=0)
                metrica_2 = st.number_input("Coperti Medi", min_value=1)
        
        submitted = st.form_submit_button("Salva Scheda")
        if submitted:
            st.success(f"✅ Anagrafica {ragione_sociale} salvata con successo!")

elif menu == "👥 Contatti":
    st.title("👥 Modulo Contatti (Rubrica)")
    st.info("Gestione dei referenti decisionali.")
    
    with st.form("nuovo_contatto"):
        col1, col2 = st.columns(2)
        with col1:
            nome = st.text_input("Nome e Cognome")
            email = st.text_input("Email")
        with col2:
            azienda = st.selectbox("Azienda/Ristorante di appartenenza", df_crm['Ragione_Sociale'].head(100)) 
            ruolo = st.text_input("Ruolo (es. IT Manager, Titolare)")
            
        submitted_contatto = st.form_submit_button("Salva Contatto")
        if submitted_contatto:
            st.success(f"✅ Contatto {nome} salvato!")

elif menu == "🤝 Offerte":
    st.title("🤝 Pipeline Commerciale")
    
    with st.form("nuova_offerta"):
        st.subheader("Crea Nuova Opportunità")
        col1, col2 = st.columns(2)
        
        with col1:
            nome_opportunita = st.text_input("Nome Opportunità")
            azienda_off = st.selectbox("Cliente / Prospect", df_crm['Ragione_Sociale'].head(100))
            prodotto_off = st.selectbox("Servizio in trattativa", list(opzioni_prodotti.values()))
            
        with col2:
            valore = st.number_input("Valore Stimato (€)", min_value=0)
            stadio = st.select_slider(
                "Stadio Pipeline", 
                options=["Qualificazione", "Demo", "Proposta Inviata", "Negoziazione", "Chiusa Vinta", "Chiusa Persa"]
            )
            
        submitted_offerta = st.form_submit_button("Salva Offerta")
        if submitted_offerta:
            st.success(f"✅ Opportunità da {valore}€ aggiornata allo stadio: {stadio}!")