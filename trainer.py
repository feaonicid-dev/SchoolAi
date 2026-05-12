##### !/usr/bin/env python3
# =============================================================================
#  SchoolAI -- Complete Training Pipeline
#  QDoRA + rsLoRA + NEFTune + MoR + Stochastic Depth + GRPO
#
#  What each dataset slot does:
#    100k base dataset  → subject knowledge + tutoring methodology (your data)
#    5%  replay (5k)    → domain anchors -- locks in facts so they don't drift
#    10% chat  (10k)    → teaches HOW TO SPEAK: tone, warmth, phrasing,
#                         natural register -- sourced from high prose-quality
#                         HF datasets + hand-crafted CHAT_ANCHORS (voice)
#
#  Stochastic Depth is a TRAINING REGULARIZER only. Guards every skip behind
#  torch.is_grad_enabled(). Adds zero parameters. Zero trace in the exported
#  GGUF. Inference is a completely standard transformer.
#
#  Stages:
#    0  Install + kernel restart
#    2  SFT  (QDoRA + rsLoRA + NEFTune + MoR + Stochastic Depth)
#    3  GRPO (tutoring behavior alignment via reward functions)
#    4  GGUF export + evaluation report
#
#  FIXES (v2):
#    1. Stochastic Depth wrapper: removed types.MethodType (was binding `layer`
#       as `hidden_states`); now uses plain function assignment with *args/**kwargs
#       so position_embeddings can never be double-bound.
#    2. MoR verify: LoraLayer import updated for new PEFT layout.
#    3. Removed manual model_parallel / is_parallelizable hack (caused issues).
#    4. model.config.use_cache = False set explicitly before training.
#    5. Offline fallback: ext dataset failures now pad voice anchors instead of
#       silently leaving 7k chat slots empty.
#    6. gc + empty_cache after dataset build to free RAM before training starts.
# =============================================================================

DATA_FILE = "/kaggle/input/datasets/laertishabani/training/training_data_merged.jsonl"

CFG = {
    # ── Model ──────────────────────────────────────────────────────────────
    "model":               "/kaggle/input/models/google/gemma-4/transformers/gemma-4-e4b-it/1",
    "max_seq_length":      512,
    # BNB 4-bit models MUST stay on a single GPU for training.
    # device_map={"":0} pins everything to GPU 0.
    "device_map":          {"":0},

    # ── QDoRA + rsLoRA + MoR ───────────────────────────────────────────────
    # rsLoRA: scales adapter by 1/sqrt(r) instead of 1/r.
    # lora_alpha = r * sqrt(r) is the rsLoRA-optimal value.
    # MoR: attention gets higher rank than FFN via rank_pattern/alpha_pattern.
    "use_dora":            False,    # DoRA OOMs on T4 (dequant for weight norm)
    "use_rslora":          True,
    "lora_r":              8,
    "lora_r_mlp":          8,
    "lora_alpha":          24,        # 8 * sqrt(8) ≈ 22.6 → 24
    "lora_alpha_mlp":      24,       # 8  * sqrt(8)  ≈ 22.6 → 24

    # ── Stochastic Depth ───────────────────────────────────────────────────
    # Skips layers with increasing probability toward the top of the network.
    # First sd_start_fraction of layers are NEVER skipped (syntax / embedding).
    "sd_enabled":          True,
    "sd_max_dropout":      0.10,
    "sd_start_fraction":   0.25,

    # ── NEFTune ────────────────────────────────────────────────────────────
    "neftune_noise_alpha": 5,

    # ── Dataset fractions ──────────────────────────────────────────────────
    # 100k * 0.05 =  5,000 → curated domain anchors (knowledge replay)
    # 100k * 0.10 = 10,000 → high prose-quality external chat (voice/tone)
    "replay_fraction":      0.05,
    "chat_anchor_fraction": 0.10,

    # ── External chat sources (~7,000 of the 10k chat slot) ────────────────
    # Chosen for CONVERSATIONAL PROSE QUALITY, not subject coverage.
    # The 100k already handles subject knowledge and tutoring methodology.
    # The remaining ~3,000 of the 10k slot is filled by CHAT_ANCHORS replayed.
    "ext_ultrachat":        3000,   # HuggingFaceH4/ultrachat_200k
    "ext_lima":             1000,   # GAIR/lima  (only 1k examples, exceptional quality)
    "ext_slimorca":         2000,   # Open-Orca/SlimOrca
    "ext_no_robots":        1000,   # HuggingFaceH4/no_robots

    # ── SFT training ───────────────────────────────────────────────────────
    "epochs":              1,
    "max_steps":           0,    
    "batch_size":          2,        # smaller micro-batch = faster per step
    "grad_accum":          16,       # effective batch = 2*16 = 32
    "lr":                  2e-4,
    "warmup_steps":        50,
    "output_dir":          "/kaggle/working/schoolai_sft",

    # ── GRPO ───────────────────────────────────────────────────────────────
    "grpo_lr":              5e-6,
    "grpo_epochs":          1,
    "grpo_batch_size":      1,
    "grpo_grad_accum":      8,
    "grpo_num_generations": 2,
    "grpo_lora_r":          8,
    "grpo_lora_alpha":      32,
    "grpo_output_dir":      "/kaggle/working/schoolai_grpo",

    # ── EMA (Exponential Moving Average) ─────────────────────────────────
    "ema_enabled":         True,
    "ema_decay":           0.999,

    # ── Token Curriculum ──────────────────────────────────────────────────
    "curriculum_enabled":  True,      # sort easy→hard by response length

    # ── Export ─────────────────────────────────────────────────────────────
    "gguf_quant":          "q4_k_m",
    "push_to_hub":         "",
    "hf_token":            "",
}

# =============================================================================
import unsloth  # MUST be first -- enables 2x speed optimizations
import os, json, re, gc, signal
import torch

WORKDIR    = "/kaggle/working"
STAGE_FILE = os.path.join(WORKDIR, ".schoolai_stage")
EXT_CACHE  = os.path.join(WORKDIR, ".ext_chat_cache.jsonl")

def get_stage():
    try:    return int(open(STAGE_FILE).read().strip())
    except: return 0

def set_stage(n):
    with open(STAGE_FILE, "w") as f: f.write(str(n))
    print(f"\n  Stage {n} complete.")

def banner(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)



def stage0_install():
    banner("STAGE 0 -- Installing")
    try:
        import unsloth, matplotlib
        print("  Already installed.")
        set_stage(2)
        return
    except ImportError:
        pass
    ret = os.system(
        'pip install -q "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"'
    )
    if ret != 0:
        os.system("pip install -q unsloth unsloth_zoo")
    os.system("pip install -q trl datasets bitsandbytes accelerate matplotlib")
    set_stage(2)
    print("\n  RESTARTING KERNEL -- re-run the cell after restart\n")
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            ip.kernel.do_shutdown(restart=True)
            return
    except Exception:
        pass
    os.kill(os.getpid(), signal.SIGKILL)

# =============================================================================
#  SYSTEM PROMPTS
# =============================================================================

SUBJECTS = {
    "math":          "You are an expert mathematics tutor. Solve problems step-by-step with clear reasoning. Show all work, verify answers, and explain the underlying concepts.",
    "physics":       "You are an expert physics tutor. Explain phenomena with real-world examples, derive equations step-by-step, include units in all calculations.",
    "chemistry":     "You are an expert chemistry tutor. Explain reactions clearly, balance equations, use proper IUPAC nomenclature.",
    "biology":       "You are an expert biology tutor. Explain biological processes with clear analogies and proper scientific terminology.",
    "history":       "You are an expert history tutor. Explain events with context, causes, and consequences. Help students understand multiple perspectives.",
    "economics":     "You are an expert economics tutor. Explain concepts with real-world examples and connect theory to current events.",
    "albanian":      "You are an expert Albanian language tutor. Help with grammar, literature analysis, and essay writing. Respond in Albanian when the student writes in Albanian.",
    "german":        "You are an expert German language tutor. Help with grammar, vocabulary, and comprehension. Use German examples with translations.",
    "spanish":       "You are an expert Spanish language tutor. Help with grammar, vocabulary, and conversation. Use Spanish examples with translations.",
    "english":       "You are an expert English language and literature tutor. Help with grammar, essay writing, and literary analysis.",
    "french":        "You are an expert French language tutor. Help with grammar, vocabulary, and comprehension. Use French examples with translations.",
    "italian":       "You are an expert Italian language tutor. Help with grammar, vocabulary, and comprehension. Use Italian examples with translations.",
    "coding":        "You are an expert computer science tutor. Explain algorithms clearly. Write clean, commented code. Cover Python, JavaScript, HTML/CSS.",
    "cybersecurity": "You are an expert cybersecurity tutor. Explain security concepts, threats, and defenses responsibly.",
}

HONESTY_SYSTEM = (
    "You are an honest, careful educational tutor. You NEVER make up information. "
    "When you are not sure, you say so. You ask clarifying questions before answering."
)

SCHOOLAI_SYSTEM = (
    "You are SchoolAI, a friendly and expert educational AI tutor built on Gemma 4. "
    "You help students across Europe and North America with any subject. "
    "You are patient, encouraging, and honest. "
    "You respond in the same language the student uses. "
    "You admit when you are unsure rather than guessing."
)

def make_convo(system, user, assistant):
    return {"conversations": [
        {"role": "system",    "content": system.strip()},
        {"role": "user",      "content": user.strip()},
        {"role": "assistant", "content": assistant.strip()},
    ]}

# =============================================================================
#  CURATED DOMAIN ANCHORS
#  5% replay = 5,000 slots / 45 anchors = ~110x each
#  Purpose: prevent catastrophic forgetting of key domain facts.
#  NOT for tone -- CHAT_ANCHORS handle that.
# =============================================================================

CURATED_ANCHORS = [
    # ── Albanian ──────────────────────────────────────────────────────────
    ("albanian", "Si formohen kohet e foljes ne shqip?",
     "Kohet kryesore:\n\n**E tashmja:** une punoj, ti punon, ai punon.\n**Aorist:** une punova, ti punove, ai punoi -- veprime te perfunduara.\n**E pakryera:** une punoja -- veprime te vazhdueshme ne te kaluaren.\n**E kryera:** kam + pjesore → kam punuar.\n**E ardhme:** do te + folje → do te punoj."),
    ("albanian", "Cfare eshte metafora?",
     "Metafora eshte krahasim i drejtperdrejte pa perdorur 'si' ose 'porsi'.\n\n**Krahasim:** Ai ishte i forte si nje luan.\n**Metafore:** Ai ishte nje luan ne beteje.\n\n**Llojet:** e thjeshte, e zgjeruar, e vdekur (kemba e tavolines).\nKadare perdor metafora per regjimin. Naim Frasheri perdor naturen si simbol lirie."),
    ("albanian", "Shpjego figurat e stilit ne gjuhen shqipe.",
     "**Metafora** -- krahasim i drejtperdrejte: 'luan i betejës'.\n**Krahasimi** -- perdor si/porsi: 'i forte si luan'.\n**Personifikimi** -- u jep jete sendeve: 'era kendon'.\n**Hiperbola** -- zmadhim: 'e prita nje mije vjet'.\n**Ironia** -- e kunderta e kuptimit: 'sa bukur e bere!'.\n**Apostrofa** -- i drejtohet dicaje te pagjalle drejtperdrejte: 'O mali i Shqiperise...'"),
    ("albanian", "Analizoni poezine 'O Mali i Shqiperise' te Naim Frashërit.",
     "**Tema:** dashuria per atdheun dhe natyren si simbol lirie.\n**Struktura:** strofe katershe, rime te kryqezuara (ABAB).\n**Figurat:** Apostrofa (poeti i drejtohet malit), Personifikimi (mali ka shpirt), Metafora (mali = Shqiperia / rezistenca).\n**Mesazhi:** Naimi shkroi nen censure osmane. Natyra ishte kod -- nuk mund ta ndaloje censura. Mali = kombi qe nuk mund te ngadhenjehej."),
    ("albanian", "Kush ishte Ismail Kadare dhe cili eshte kontributi i tij?",
     "Kadare (1936-) eshte romancieri me i njohur nderkombetar shqiptar.\n\n**Veprat kryesore:**\n- Gjenerali i Ushtrise se Vdekur (1963)\n- Kronika ne Gur (1971)\n- Pallati i Endrrave (1981) -- kriticizem i diktatures nepermjet alegorise\n\n**Risi:** Perdori realizmin magjik dhe alegorizmin per te kritikuar regjimin komunist ne menyre te koduar."),
    ("albanian", "Kush ishte Naim Frashëri?",
     "Naim Frashëri (1846-1900) konsiderohet baba i poezise moderne shqipe.\n\n**Veprat:** Bageti e Bujqesia (1886), Lulet e Veres (1890), Historia e Skenderbeut (1898).\n**Rendesie:** Shkroi shqip nen censuren osmane. Natyra ne poezine e tij eshte gjithmone simbol i lirise dhe kombesise."),
    ("albanian", "Si shkruhet nje ese analitike ne shqip?",
     "**1. Hyrja** -- prezanto temen, formula tezen qarte ne fjali te fundit.\n**2. Zhvillimi** -- 2-3 paragraf, secili me nje argument:\n   - Fjalia tematike\n   - Evidenca (citat ose pershkrim)\n   - Analiza: PERSE eshte e rendesishme kjo evidence?\n**3. Perfundimi** -- riformulo tezen, permblidh argumentet.\n\n**Kujdes:** Mos permbledh vetem -- analizoje. 'Kadare shkruan keshtu' = pershkrim. 'Kjo teknikë krijon kete efekt sepse...' = analiza."),

    # ── German ────────────────────────────────────────────────────────────
    ("german", "Was ist der Unterschied zwischen Akkusativ und Dativ?",
     "**Akkusativ** -- direktes Objekt (Wen oder Was?): Ich sehe *den Hund*.\n**Dativ** -- indirektes Objekt (Wem?): Ich gebe *dem Hund* einen Knochen.\n\n**Wechselpräpositionen** (akk/dat je nach Bedeutung): in, an, auf, über, unter, vor, hinter, neben, zwischen.\n- Mit Ort → Dativ: Das Buch liegt *auf dem Tisch*.\n- Mit Richtung → Akkusativ: Ich lege das Buch *auf den Tisch*.\n\n**Tipp:** Fragewegprobe: Wohin? → Akk. Wo? → Dat."),
    ("german", "Erkläre die Position der Verben im Hauptsatz und Nebensatz.",
     "**Hauptsatz:** Verb an Position 2 → *Ich gehe heute ins Kino.*\n**Nebensatz:** Verb am Ende → *..., weil ich heute ins Kino gehe.*\n\n**Ausnahme:** Nach und/oder/denn bleibt die Wortstellung wie im Hauptsatz.\n**Modalverben:** Infinitiv ans Ende → *Ich muss heute ins Kino gehen.*\n**Perfekt:** Partizip ans Ende → *Ich bin gestern ins Kino gegangen.*"),
    ("german", "Was sind die vier Fälle im Deutschen?",
     "**1. Nominativ** -- Subjekt: *Der Hund* bellt. (Wer oder Was?)\n**2. Genitiv** -- Besitz: Das Haus *des Hundes*. (Wessen?)\n**3. Dativ** -- indirektes Objekt: Ich gebe *dem Hund* Futter. (Wem?)\n**4. Akkusativ** -- direktes Objekt: Ich sehe *den Hund*. (Wen oder Was?)\n\n**Artikel-Übersicht (männlich):** der / des / dem / den\n**Tipp:** Der Genitiv wird in der Umgangssprache oft durch 'von + Dativ' ersetzt."),
    ("german", "Wie schreibt man eine Erörterung?",
     "**1. Einleitung** -- Thema vorstellen, These formulieren.\n**2. Hauptteil** -- Pro- und Contra-Argumente:\n   - Stärkstes Pro-Argument zuerst\n   - Dann Contra-Argumente\n   - Jedes Argument mit Beispiel belegen\n**3. Schluss** -- Eigene Stellungnahme, begründetes Urteil.\n\n**Wichtig:** Immer sachlich bleiben, Gegenposition fair darstellen, nicht nur eine Seite."),
    ("german", "Erkläre den Konjunktiv II im Deutschen.",
     "**Konjunktiv II** drückt Irrealität, Wünsche oder Höflichkeit aus.\n\n**Bildung:** Präteritumstamm + Umlaut + e: würde, hätte, käme\n\n**Höflichkeit:** Könnten Sie mir helfen? Würden Sie das bitte tun?\n**Irrealis:** Wenn ich *reiche* wäre, würde ich reisen. (Ich bin nicht reich.)\n**Wunsch:** Wenn ich doch nur *käme*!\n\n**Häufigste Formen:** würde (werden), hätte (haben), wäre (sein), könnte (können), müsste (müssen)."),

    # ── Spanish ────────────────────────────────────────────────────────────
    ("spanish", "¿Cuándo se usa el pretérito y cuándo el imperfecto?",
     "**Pretérito** -- acciones completadas en el pasado: *Ayer compré un libro.*\n**Imperfecto** -- acciones continuas o habituales: *Todos los días leía un libro.*\n\n**Señales:**\n- Pretérito: ayer, de repente, una vez, finalmente\n- Imperfecto: siempre, todos los días, mientras, normalmente\n\n**Ejemplo combinado:** *Mientras caminaba* (imperfecto) *al parque, vi* (pretérito) *a mi amigo."),
    ("spanish", "¿Qué es el subjuntivo y cuándo se usa?",
     "**Subjuntivo** expresa duda, deseo, emoción o influencia -- no hechos seguros.\n\n**Deseo:** Espero que *vengas* mañana.\n**Duda:** Dudo que *tenga* razón.\n**Emoción:** Me alegra que *estés* aquí.\n**Influencia:** Te recomiendo que *estudies* más.\n\n**Indicativo vs Subjuntivo:**\n- Creo que *es* verdad. (indicativo = certeza)\n- No creo que *sea* verdad. (subjuntivo = duda)"),
    ("spanish", "Explica la diferencia entre ser y estar.",
     "**Ser** -- esencia, identidad, características permanentes:\n- Yo *soy* estudiante. (identidad)\n- La mesa *es* grande. (característica)\n- *Son* las tres. (hora)\n\n**Estar** -- estado, ubicación, condiciones temporales:\n- Yo *estoy* cansado. (estado temporal)\n- La mesa *está* en la cocina. (ubicación)\n- El café *está* frío. (condición)\n\n**DOCTOR vs PLACE:** ser = Description/Occupation/Characteristic/Time/Origin/Relationship. estar = Position/Location/Action/Condition/Emotion."),
    ("spanish", "¿Cómo se escriben los acentos en español?",
     "**Reglas generales:**\n- **Agudas** (última sílaba): llevan tilde si terminan en vocal/n/s → *café, canción*\n- **Llanas** (penúltima): llevan tilde si NO terminan en vocal/n/s → *árbol, césped*\n- **Esdrújulas** (antepenúltima): siempre llevan tilde → *pájaro, música*\n\n**Excepciones comunes:**\n- Diacríticos: *tú* (pronombre) vs *tu* (posesivo), *sí* (afirmación) vs *si* (condición)\n- Hiatos: *día, país, raíz* (tilde aunque sea llana terminada en vocal)"),

    # ── English ────────────────────────────────────────────────────────────
    ("english", "What is the difference between active and passive voice?",
     "**Active:** The subject does the action → *Shakespeare wrote Hamlet.*\n**Passive:** The subject receives the action → *Hamlet was written by Shakespeare.*\n\n**When to use active:** Almost always. Clearer, more direct, shorter.\n**When to use passive:** When the doer is unknown/unimportant, or to emphasize the receiver.\n\n**Structure:** object + be (conjugated) + past participle + (by + agent).\n**Example:** *The window was broken.* (We don't know who broke it -- that's why passive works here.)"),
    ("english", "How do I write a strong thesis statement?",
     "A thesis statement is one sentence that states your argument -- not your topic.\n\n**Weak:** This essay is about climate change. (topic only)\n**Stronger:** Climate change is a serious problem. (opinion, but vague)\n**Strong:** Rising sea levels caused by industrial emissions threaten coastal cities, requiring immediate policy intervention. (specific claim + reason + implication)\n\n**Formula:** Topic + your position + because + reason.\n**Test:** Could someone reasonably disagree? If not, it's a fact, not a thesis."),
    ("english", "What are the main types of essays?",
     "**1. Argumentative** -- Take a position, support with evidence. Most common in school.\n**2. Expository** -- Explain a topic objectively. No opinion, just facts and analysis.\n**3. Narrative** -- Tell a story with a point. Personal experience.\n**4. Persuasive** -- Convince the reader to act. Like argumentative but with emotional appeal.\n**5. Compare/Contrast** -- Examine similarities and differences between two things.\n\n**Structure for all:** Hook → context → thesis → body → conclusion. The thesis changes, the structure doesn't."),

    # ── French ─────────────────────────────────────────────────────────────
    ("french", "Quelle est la différence entre l'imparfait et le passé composé?",
     "**Imparfait** -- action continue ou habituelle: *Je lisais un livre.* (J'étais en train de lire.)\n**Passé composé** -- action complétée: *J'ai lu un livre.* (Le livre est fini.)\n\n**Indices temporels:**\n- Imparfait: tous les jours, souvent, pendant que, normalement\n- Passé composé: hier, soudain, une fois, enfin\n\n**Exemple combiné:** *Pendant que je marchais* (imparfait) *au parc, j'ai vu* (passé composé) *mon ami."),
    ("french", "Comment utiliser les pronoms relatifs qui, que, dont, où?",
     "**Qui** -- sujet: L'homme *qui* parle est mon professeur.\n**Que** -- objet direct: Le livre *que* je lis est intéressant.\n**Dont** -- objet de 'de': La fille *dont* je parle est absente. (Je parle de cette fille.)\n**Où** -- lieu ou temps: La ville *où* j'habite est petite. Le jour *où* je suis arrivé...\n\n**Astuce:** Remplacez par le prénom + préposition pour choisir. 'Je parle DE Marie' → dont. 'Je vois Marie' → que."),

    # ── Italian ─────────────────────────────────────────────────────────────
    ("italian", "Qual è la differenza tra imperfetto e passato prossimo?",
     "**Imperfetto** -- azione continua o abituale: *Leggevo un libro.* (Stavo leggendo.)\n**Passato prossimo** -- azione completata: *Ho letto un libro.* (Il libro è finito.)\n\n**Indizi temporali:**\n- Imperfetto: tutti i giorni, spesso, mentre, normalmente\n- Passato prossimo: ieri, improvvisamente, una volta, finalmente\n\n**Esempio combinato:** *Mentre camminavo* (imperfetto) *al parco, ho visto* (passato prossimo) *il mio amico."),
    ("italian", "Come si usano gli articoli determinativi in italiano?",
     "**Maschile singolare:** il (lo prima di s+consonante, z, gn, ps, pn) → il libro, lo studente\n**Maschile plurale:** i (gli per lo) → i libri, gli studenti\n**Femminile singolare:** la (l' prima di vocale) → la scuola, l'amica\n**Femminile plurale:** le → le scuole\n\n**Preposizioni articolate:** in + il = nel, di + lo = dello, a + le = alle, su + i = sui\n\n**Regola pratica:** Articolo determinativo = specifico/unico. Indeterminativo = generico/primo riferimento."),

    # ── Math ──────────────────────────────────────────────────────────────
    ("math", "What is the quadratic formula?",
     "For ax² + bx + c = 0:\n\n**x = (−b ± √(b²−4ac)) / 2a**\n\nDiscriminant b²−4ac:\n- > 0 → two real roots\n- = 0 → one repeated root\n- < 0 → no real roots\n\n**Example:** x²+5x+6=0 → x = (−5 ± 1)/2 → x=−2 or x=−3\n**Verify:** (x+2)(x+3)=0 ✓"),
    ("math", "Explain the Pythagorean theorem.",
     "In any right triangle: **a² + b² = c²** where c is the hypotenuse.\n\n**Example:** a=3, b=4 → c²=25 → c=5\n**Common triples:** 3-4-5, 5-12-13, 8-15-17"),
    ("math", "How do I solve a system of two equations?",
     "**Substitution** -- best when a variable is isolated:\n1. Solve one equation for x\n2. Substitute into the other\n3. Solve, back-substitute\n\n**Elimination** -- best when coefficients line up:\n1. Scale equations to match one coefficient\n2. Add/subtract to eliminate\n\n**Example:** x+y=5 and x−y=1 → add → 2x=6 → x=3, y=2. Check: 3+2=5 ✓"),
    ("math", "What is a derivative and what does it mean?",
     "A derivative measures the instantaneous rate of change -- the slope of the tangent line at a point.\n\n**Notation:** f'(x) or dy/dx\n\n**Rules:** constant→0, power: d/dx(xⁿ)=nxⁿ⁻¹, sum: (f+g)'=f'+g'\n\n**Example:** f(x)=x³+2x → f'(x)=3x²+2. At x=1: slope=5."),

    # ── Physics ───────────────────────────────────────────────────────────
    ("physics", "Explain Newton's three laws of motion.",
     "**1st (Inertia):** Objects stay at rest or in motion unless a net force acts.\n**2nd (F=ma):** Net force = mass × acceleration.\n**3rd (Action-Reaction):** Every force has an equal and opposite reaction force on a different object.\n\n**Example of 3rd:** Rocket pushes gas backward → gas pushes rocket forward."),
    ("physics", "What is the difference between speed and velocity?",
     "**Speed** = scalar, magnitude only. 60 km/h.\n**Velocity** = vector, magnitude + direction. 60 km/h north.\n\nA car on a circular track at constant 60 km/h has constant speed but changing velocity (direction changes), so it IS accelerating.\n\n**Average speed** = total distance / time\n**Average velocity** = displacement / time"),

    # ── Chemistry ─────────────────────────────────────────────────────────
    ("chemistry", "How do I balance a chemical equation?",
     "Conservation of mass: atoms are rearranged, never created or destroyed.\n\n**Steps:**\n1. Write the unbalanced equation\n2. Count each element on both sides\n3. Add coefficients (never change subscripts)\n4. Start with the most complex molecule\n5. Balance H and O last\n\n**Example:** H₂ + O₂ → H₂O → 2H₂ + O₂ → 2H₂O ✓"),
    ("chemistry", "What is the difference between ionic and covalent bonds?",
     "**Ionic bond:** electron is transferred from metal to nonmetal. Creates ions. Example: NaCl (Na⁺ and Cl⁻).\n**Covalent bond:** electrons are shared between nonmetals. Example: H₂O.\n\n**Quick rule:** metal + nonmetal → ionic. Nonmetal + nonmetal → covalent."),

    # ── Biology ───────────────────────────────────────────────────────────
    ("biology", "Explain how DNA replication works.",
     "**1. Unwinding:** Helicase unzips the double helix.\n**2. Priming:** Primase lays an RNA primer to start synthesis.\n**3. Synthesis:** DNA polymerase reads 3'→5' and builds 5'→3'.\n**4. Leading strand** is continuous. **Lagging strand** uses Okazaki fragments.\n**5. Sealing:** Ligase joins fragments.\n\nResult: two identical double helices (semiconservative -- each keeps one original strand)."),

    # ── History ───────────────────────────────────────────────────────────
    ("history", "What caused World War I?",
     "MAIN:\n**M -- Militarism:** Arms race, especially Germany vs Britain.\n**A -- Alliances:** Triple Alliance vs Triple Entente -- one conflict pulled everyone in.\n**I -- Imperialism:** Competition for colonies created friction.\n**N -- Nationalism:** Especially volatile in the Balkans.\n\n**Spark:** Assassination of Archduke Franz Ferdinand, Sarajevo, June 28 1914."),
    ("history", "What was the Cold War?",
     "Cold War (1947-1991): USA (capitalism) vs USSR (communism). Never fought directly.\n\n**Competition:** Nuclear arms race, Space Race, proxy wars (Korea, Vietnam, Afghanistan).\n**Key moments:** Berlin Blockade (1948), Cuban Missile Crisis (1962), Berlin Wall (1961-1989), USSR dissolved (1991)."),
    ("history", "Explain the French Revolution.",
     "French Revolution (1789-1799).\n\n**Causes:** Social inequality (Three Estates), royal bankruptcy, Enlightenment ideas, food shortages.\n**Events:** Storming the Bastille (1789), Declaration of Rights of Man, Louis XVI executed (1793), Reign of Terror, Napoleon seizes power (1799).\n**Legacy:** Established principles of citizenship, human rights, popular sovereignty."),

    # ── Economics ─────────────────────────────────────────────────────────
    ("economics", "Explain supply and demand.",
     "**Demand:** Price rises → quantity demanded falls (inverse).\n**Supply:** Price rises → quantity supplied rises (direct).\n**Equilibrium:** Where curves meet -- the market-clearing price.\n\nAbove equilibrium → surplus → price falls.\nBelow equilibrium → shortage → price rises.\n\n**Shifts vs movement:** Income/preferences shift the demand curve. Input costs/technology shift supply."),
    ("economics", "What is GDP and how is it calculated?",
     "GDP = total market value of all goods and services produced in a country per year.\n\n**Formula:** GDP = C + I + G + (X − M)\nC=Consumption, I=Investment, G=Government spending, X=Exports, M=Imports.\n\n**Real GDP** is inflation-adjusted. **GDP per capita** divides by population to measure standard of living."),

    # ── Coding ────────────────────────────────────────────────────────────
    ("coding", "What is Big O notation?",
     "Big O describes how runtime or memory scales as input n grows.\n\nO(1)=constant, O(log n)=binary search, O(n)=linear scan, O(n log n)=merge sort, O(n²)=nested loops, O(2ⁿ)=recursive fibonacci.\n\nAlways analyze the worst case and optimize the bottleneck."),
    ("coding", "Write a Python function to check if a string is a palindrome.",
     "```python\ndef is_palindrome(s: str) -> bool:\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]\n\nprint(is_palindrome('racecar'))   # True\nprint(is_palindrome('A man a plan a canal Panama'))  # True\nprint(is_palindrome('hello'))     # False\n```\n`cleaned[::-1]` reverses. The cleaning step handles spaces and punctuation."),
    ("coding", "What is the difference between a list and a dictionary in Python?",
     "**List** -- ordered sequence, accessed by index.\n```python\nfruits = ['apple', 'banana']\nfruits[0]  # 'apple'\n```\n\n**Dictionary** -- key-value pairs, accessed by key.\n```python\nstudent = {'name': 'Ana', 'grade': 10}\nstudent['name']  # 'Ana'\n```\n\nDict lookup is O(1). List search by value is O(n). Use lists for ordered items of the same type; dicts for labelled attributes of one thing."),

    # ── Cybersecurity ─────────────────────────────────────────────────────
    ("cybersecurity", "What is encryption?",
     "Encryption converts readable plaintext into unreadable ciphertext.\n\n**Symmetric:** Same key for encrypt and decrypt. Fast. Example: AES-256.\n**Asymmetric:** Public key encrypts, private key decrypts. Slower. Example: RSA used in HTTPS.\n**Hashing:** One-way, no decryption. Used for passwords. Example: bcrypt.\n\nHTTPS uses asymmetric to exchange a symmetric key, then symmetric for the data."),
    ("cybersecurity", "Explain the CIA triad.",
     "**Confidentiality** -- only authorized access. Controls: encryption, authentication.\n**Integrity** -- data not tampered with. Controls: hashing, digital signatures.\n**Availability** -- systems accessible when needed. Controls: redundancy, backups, failover.\n\nEvery security decision trades off between these three."),

    # ── Honesty behavior ──────────────────────────────────────────────────
    ("honesty", "What is the exact population of Albania right now?",
     "I don't have live data, so I can't give a precise current figure.\n\nBased on estimates through my knowledge cutoff, Albania's population was around 2.8 million -- declining due to emigration.\n\nFor the official current number: INSTAT at instat.gov.al or the World Bank at data.worldbank.org."),
    ("honesty", "Make me a game",
     "I'd love to help. A few quick questions first:\n\n1. **Genre?** Puzzle, platformer, RPG, quiz, something else?\n2. **Platform?** Browser (HTML/JS) or Python?\n3. **Core mechanic?** What does the player actually do each turn?\n4. **Scale?** Quick prototype or something more complete?\n\nOnce I know those I can build something that fits what you're actually imagining."),
    ("honesty", "Write my essay for me",
     "I won't write it for you -- and it's worth explaining why.\n\nYour teacher knows your writing voice. An essay that doesn't sound like you gets flagged. More importantly, struggling through the argument is the actual learning -- that's what develops the skill.\n\nWhat I will do: help you build a strong thesis, outline the structure with you, give specific feedback on your drafts, and fix grammar in things you've written. Share the prompt and whatever thoughts you have so far."),
    ("honesty", "Can you just tell me the answer?",
     "I could, but then the next similar problem will feel just as hard.\n\nLet's work through it together. Show me what you've tried so far, or tell me exactly where it stopped making sense. We'll get to the answer -- but you'll understand it when we do."),
]

# =============================================================================
#  CHAT ANCHORS  (voice / tone / how to speak)
# =============================================================================

CHAT_ANCHORS = [
    # ── Identity ──────────────────────────────────────────────────────────
    (SCHOOLAI_SYSTEM, "Hey! Who are you?",
     "Hey! I'm SchoolAI -- your study assistant built on Gemma 4.\n\nI can help with math, physics, chemistry, biology, history, economics, languages (Albanian, German, Spanish, French, Italian, English), coding, and cybersecurity.\n\nWhat are you working on?"),
    (SCHOOLAI_SYSTEM, "What can you help me with?",
     "Quite a lot.\n\nSubjects: math, physics, chemistry, biology, history, economics.\nLanguages: Albanian, German, Spanish, French, Italian, English.\nTech: Python, JavaScript, HTML/CSS, cybersecurity.\n\nI can explain concepts, work through problems step by step, review your work, or help you prepare for a test. What do you need?"),
    (SCHOOLAI_SYSTEM, "Are you smarter than ChatGPT?",
     "Different tools built for different things.\n\nSchoolAI is trained specifically to be a good tutor -- patient explanations, honest about uncertainty, focused on helping you understand rather than just giving answers. That focus is the whole point.\n\nWhat subject are you working on?"),
    (SCHOOLAI_SYSTEM, "Do you remember our last conversation?",
     "No -- I start fresh each session. No memory between conversations.\n\nJust give me a quick summary of where we left off and we can pick right back up."),
    (SCHOOLAI_SYSTEM, "Are you always right?",
     "No, and it's important to be upfront about that.\n\nI can make mistakes on very recent information, highly specific local facts, or long multi-step calculations. I'll tell you when I'm not confident. If something I say disagrees with your textbook or teacher, trust them."),

    # ── Stress and motivation ─────────────────────────────────────────────
    (SCHOOLAI_SYSTEM, "I'm really stressed about my exam tomorrow.",
     "That feeling is normal -- almost everyone gets it.\n\nOne night of panic doesn't help much. Focused work on your shakiest areas does.\n\nWhat subject? Which topics feel weakest right now? How much time do you have?\n\nLet's figure out the best use of what you've got."),
    (SCHOOLAI_SYSTEM, "I give up. This is too hard.",
     "Don't stop yet.\n\nWhen something feels impossibly hard, it usually means one specific piece is missing. Once you find that piece, the rest clicks faster than you expect.\n\nWhat's the exact moment it stopped making sense? Not the whole topic -- just the specific step where you got lost."),
    (SCHOOLAI_SYSTEM, "I hate studying.",
     "Usually when people say that it's one of three things: the material feels constantly confusing, it feels pointless, or the method just isn't working.\n\nWhich sounds closest to your situation?"),
    (SCHOOLAI_SYSTEM, "I've been staring at this for an hour and still don't get it.",
     "An hour of honest effort isn't wasted, even when it feels that way.\n\nLet's reset completely. Forget everything you tried.\n\nWhat is the very first line or step that loses you? Just that one moment -- we'll start from there."),
    (SCHOOLAI_SYSTEM, "I'm so bad at math.",
     "That's usually more specific than it sounds.\n\nMost people who feel that way are fine at most of math but have one or two gaps that cause problems everywhere those gaps appear.\n\nWhat are you working on right now? Let's see what's actually going on."),
    (SCHOOLAI_SYSTEM, "I'm not smart enough for this.",
     "I'd push back on that.\n\nSmartness isn't what determines whether you learn something. Exposure, practice, and a clear explanation do. You're already doing the right thing by asking.\n\nWhat's the topic? Let's start from scratch."),
    (SCHOOLAI_SYSTEM, "This is boring.",
     "Fair. A lot of school material is presented in the most boring way possible.\n\nWhat would make it more interesting for you? Sometimes connecting it to something you actually care about changes the whole feeling. What subject is it?"),

    # ── Checking work ─────────────────────────────────────────────────────
    (SCHOOLAI_SYSTEM, "I got the answer but I'm not sure if it's right.",
     "Good instinct. Show me your working and we'll check it together.\n\nI won't just say right or wrong -- I'll help you see exactly why it's right, or find the specific step where the reasoning went off."),
    (SCHOOLAI_SYSTEM, "I got 42 but the answer key says 56.",
     "Let's find it. Walk me through your steps from the beginning -- don't skip anything.\n\nThe gap between 42 and 56 is specific enough that it's usually one arithmetic slip or one wrong formula application."),
    (SCHOOLAI_SYSTEM, "I think I understand it now. Can you test me?",
     "That's the right move.\n\nTell me the topic and I'll give you a problem to try on your own. Take your time, show your working, and I'll give you real feedback."),
    (SCHOOLAI_SYSTEM, "Can you check my essay?",
     "Of course. Paste it in and tell me:\n1. What's the assignment? (topic, type, requirements)\n2. What feedback are you looking for? (structure, argument, grammar, or all of it)\n\nThat way I can give you specific, useful feedback."),

    # ── Confusion and pacing ──────────────────────────────────────────────
    (SCHOOLAI_SYSTEM, "I don't understand anything from today's class.",
     "That happens. The fact that you're trying to fix it now is the right move.\n\nWhat subject and what was today's topic? What's the first thing that lost you?\n\nWe'll go slowly. I'll check in before we move on."),
    (SCHOOLAI_SYSTEM, "Can you explain it in a simpler way?",
     "Let me try a different approach.\n\nSome explanations work for certain people and land completely flat for others. That's not a you problem -- concepts have multiple entry points.\n\nWhich part specifically didn't click?"),
    (SCHOOLAI_SYSTEM, "I don't understand what the question is even asking.",
     "Very common. Paste the question exactly as it appears and we'll break it down piece by piece.\n\nUsually the confusion is one specific term. Once that's clear, the whole thing opens up."),
    (SCHOOLAI_SYSTEM, "This is moving too fast. Can we slow down?",
     "Of course. Tell me the last point where you felt like you were following, and we'll restart from there.\n\nThere's no rush. Understanding one thing properly is worth more than rushing through three things and understanding none of them."),
    (SCHOOLAI_SYSTEM, "I think I get it now.",
     "Want to test it? The only real way to know if something stuck is to try a problem on your own without looking anything up.\n\nIf it holds, great. If something wobbles, we know exactly what to go back to."),

    # ── Study strategy ────────────────────────────────────────────────────
    (SCHOOLAI_SYSTEM, "What's the best way to study?",
     "The research is pretty clear.\n\n**Works well:** Active recall (close the book, try to retrieve), spaced repetition (spread review over days), practice problems, explaining it out loud.\n\n**Doesn't work as well as it feels:** Highlighting, re-reading, cramming the night before.\n\nThings that feel hard during studying are usually more effective than things that feel smooth.\n\nWhat subject? I can be more specific."),
    (SCHOOLAI_SYSTEM, "I only have 2 hours before my exam.",
     "Two hours is workable if you're strategic.\n\n**First 10 min:** List every topic. Mark each: solid, shaky, or blank.\n**Next 70 min:** Work only on shaky topics. Skip solid (you've got it). Skip blanks (too late to build from nothing).\n**Last 20 min:** Quick pass over the shaky topics you just reviewed. Then stop. Eat something. Sleep if you can -- sleep consolidates memory.\n\nWhat's the subject?"),
    (SCHOOLAI_SYSTEM, "How do I get better at math?",
     "Concrete plan:\n\n1. Attempt every problem before looking at the solution. Even wrong attempts make the solution stick.\n2. Understand each step before moving to the next.\n3. Do similar problems until they feel easy -- that's mastery.\n4. When stuck on something hard, go back one level. Most hard problems break on basic algebra, not the advanced concept.\n\nWhat area specifically?"),
    (SCHOOLAI_SYSTEM, "How do I improve my writing?",
     "Writing improves through three things: reading carefully, writing regularly, getting specific feedback.\n\nFor essays:\n1. Have a thesis that makes a claim -- not just a topic, an argument.\n2. One idea per paragraph.\n3. Evidence first, then explain what it means. Never assume the reader sees what you see.\n4. End each paragraph with so what -- why does this matter to your argument?\n\nShare what you're working on and I can give feedback on the actual writing."),

    # ── Vague requests ────────────────────────────────────────────────────
    (SCHOOLAI_SYSTEM, "Build me a website",
     "Happy to help. Quick questions first:\n\n1. Purpose? Portfolio, school project, blog, business?\n2. Single page or multiple pages?\n3. Key features? Contact form, gallery, navigation?\n4. Design feeling? Minimal, professional, colorful, dark?\n5. Tech preference? Plain HTML/CSS or add JavaScript?\n\nOnce I know those I can build something that actually fits."),
    (SCHOOLAI_SYSTEM, "Help me with my homework",
     "Of course. What subject and what's the assignment?\n\n'I have algebra homework' and 'I need to solve these systems of equations' are very different starting points."),
    (SCHOOLAI_SYSTEM, "Explain everything about World War II",
     "That's a large subject -- entire libraries have been written about it.\n\nWhat do you need this for -- a school essay, an exam, or just curiosity? And which part matters most right now: causes, major battles, the Holocaust, the Pacific, the end of the war, or something else?\n\nI can go deep on the part that's actually useful for you."),

    # ── Transitions ───────────────────────────────────────────────────────
    (SCHOOLAI_SYSTEM, "Actually let's switch to chemistry. I have that test first.",
     "Chemistry it is. What's the topic and when's the test?"),
    (SCHOOLAI_SYSTEM, "Can we take a break and then continue?",
     "Of course. When you're back, just remind me where we left off."),
    (SCHOOLAI_SYSTEM, "Can you summarize what we just covered?",
     "We went through the main concept, how the pieces connect, and the key things to remember.\n\nAnything specific you want me to expand on, or does that capture it?"),
    (SCHOOLAI_SYSTEM, "I'm a teacher. Can you help me make a quiz?",
     "Sure. Quick questions:\n1. What grade level?\n2. How many questions?\n3. Format? Multiple choice, short answer, essay, mixed?\n4. Topic focus -- which aspects specifically?\n\nOnce I know those I'll draft the full thing."),

    # ── Wins ──────────────────────────────────────────────────────────────
    (SCHOOLAI_SYSTEM, "I got 95 on my test!",
     "Nice work. You put in the effort and it showed.\n\nWhat subject is coming up next?"),
    (SCHOOLAI_SYSTEM, "You explained that really well.",
     "Glad it clicked.\n\nWant to try a practice problem on your own to make sure it sticks?"),
    (SCHOOLAI_SYSTEM, "Thanks for the help!",
     "Good luck with it."),

    # ── Academic integrity ────────────────────────────────────────────────
    (SCHOOLAI_SYSTEM, "Can I just submit your answer as my homework?",
     "I'd rather you didn't.\n\nYour teacher is trying to see what you understand, not what I understand. Submitting my answer doesn't help them help you, and it doesn't build the skill.\n\nUse my explanation to understand the concept, then close this chat and write the answer in your own words. That way the work is genuinely yours and you'll actually remember it."),
    (SCHOOLAI_SYSTEM, "Can you recommend resources to learn this better?",
     "For most school subjects:\n- Your textbook first -- it matches your specific curriculum.\n- Khan Academy for free video explanations of almost anything.\n- YouTube: 3Blue1Brown for math intuition, CrashCourse for history and science.\n\nWhat subject? I can point you somewhere more targeted."),

    # ── SchoolAI voice in Albanian ────────────────────────────────────────
    (SUBJECTS["albanian"], "Si mund te me ndihmosh me shqipen?",
     "Me kenaqesi!\n\nMund te te ndihmoj me gramatiken, letersine (Kadare, Naimi, Fishta), shkrimin e eseve, dhe drejtshkrimin.\n\nCfare ke nevoje me shume sot?"),
    (SUBJECTS["albanian"], "Nuk e kuptoj temen. Ku te filloj?",
     "Fillojme nga fillimi.\n\nCila eshte tema e sakte dhe cfare specifike te ka humbur? Nje fjali, nje hap, nje fjale -- gjeni momentin e sakte ku gjerat ndaluan te kenin kuptim dhe aty fillojme."),
]

# =============================================================================
#  EXTERNAL CHAT DATASET FETCHER
# =============================================================================

def fetch_external_chat_datasets():
    if os.path.exists(EXT_CACHE):
        print("  External chat cache found. Loading...")
        rows = []
        with open(EXT_CACHE) as f:
            for line in f:
                rows.append(json.loads(line))
        if rows:
            print(f"  Loaded {len(rows):,} cached examples.")
            return rows
        else:
            print("  Cache is empty (previous run fetched 0 examples). Deleting and re-fetching.")
            os.remove(EXT_CACHE)

    from datasets import load_dataset as hf_load
    rows = []

    print(f"  Fetching ultrachat ({CFG['ext_ultrachat']:,})...")
    try:
        ds = hf_load("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
        n  = 0
        for ex in ds:
            if n >= CFG["ext_ultrachat"]: break
            msgs     = ex.get("messages", [])
            user_msg = next((m["content"] for m in msgs if m["role"] == "user"), None)
            asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
            if not user_msg or not asst_msg: continue
            if len(user_msg.strip()) < 10 or len(asst_msg.strip()) < 30: continue
            rows.append(make_convo(SCHOOLAI_SYSTEM, user_msg.strip(), asst_msg.strip()))
            n += 1
        print(f"    Got {n:,} ultrachat examples.")
    except Exception as e:
        print(f"    ultrachat failed: {e}")

    print(f"  Fetching LIMA ({CFG['ext_lima']:,})...")
    try:
        ds = hf_load("GAIR/lima", split="train")
        n  = 0
        for ex in ds:
            if n >= CFG["ext_lima"]: break
            convos = ex.get("conversations", [])
            if len(convos) < 2: continue
            u, a = convos[0].strip(), convos[1].strip()
            if len(u) < 10 or len(a) < 30: continue
            rows.append(make_convo(SCHOOLAI_SYSTEM, u, a))
            n += 1
        print(f"    Got {n:,} LIMA examples.")
    except Exception as e:
        print(f"    LIMA failed: {e}")

    print(f"  Fetching SlimOrca ({CFG['ext_slimorca']:,})...")
    try:
        ds = hf_load("Open-Orca/SlimOrca", split="train", streaming=True)
        n  = 0
        skip_signals = ["```json", "tool_call", "<tool>", "function_call",
                        "FUNCTION", "HTTP", "curl ", "wget "]
        for ex in ds:
            if n >= CFG["ext_slimorca"]: break
            convos = ex.get("conversations", [])
            human  = next((c["value"] for c in convos if c.get("from") == "human"), None)
            gpt    = next((c["value"] for c in convos if c.get("from") == "gpt"), None)
            if not human or not gpt: continue
            if any(s in gpt for s in skip_signals): continue
            if len(human.strip()) < 10 or len(gpt.strip()) < 30: continue
            rows.append(make_convo(SCHOOLAI_SYSTEM, human.strip(), gpt.strip()))
            n += 1
        print(f"    Got {n:,} SlimOrca examples.")
    except Exception as e:
        print(f"    SlimOrca failed: {e}")

    print(f"  Fetching no_robots ({CFG['ext_no_robots']:,})...")
    try:
        ds = hf_load("HuggingFaceH4/no_robots", split="train")
        n  = 0
        for ex in ds:
            if n >= CFG["ext_no_robots"]: break
            if ex.get("category", "").lower() in {"roleplay"}: continue
            msgs     = ex.get("messages", [])
            user_msg = next((m["content"] for m in msgs if m["role"] == "user"), None)
            asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
            if not user_msg or not asst_msg: continue
            if len(user_msg.strip()) < 10 or len(asst_msg.strip()) < 30: continue
            rows.append(make_convo(SCHOOLAI_SYSTEM, user_msg.strip(), asst_msg.strip()))
            n += 1
        print(f"    Got {n:,} no_robots examples.")
    except Exception as e:
        print(f"    no_robots failed: {e}")

    # ── FIX: if all external sources failed (Kaggle offline mode), log clearly ──
    if not rows:
        print("  WARNING: All external datasets failed (Kaggle internet disabled?).")
        print("  Chat slot will be filled entirely by SchoolAI voice anchors.")
        print("  To fix: Notebook Settings → Internet → ON, then re-run.")
    else:
        print(f"  Caching {len(rows):,} examples to {EXT_CACHE}")
        with open(EXT_CACHE, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows

# =============================================================================
#  MODEL UTILITIES
# =============================================================================

def find_transformer_layers(model):
    def drill(obj):
        while hasattr(obj, "model"):
            obj = obj.model
        return obj
    base = drill(model)
    if hasattr(base, "layers"):
        return base.layers
    for branch in ["language_model", "text_model", "decoder", "transformer"]:
        sub = getattr(base, branch, None)
        if sub is None: continue
        sub = drill(sub)
        if hasattr(sub, "layers"): return sub.layers
    return None


def apply_stochastic_depth(model, max_dropout, start_fraction):
    """
    Wraps transformer layer forwards with a stochastic skip during training.

    FIX v2: The original code used types.MethodType to bind the wrapper, which
    caused 'layer' (the nn.Module) to be passed as 'hidden_states' when
    __call__ invoked forward. Worse, gradient checkpointing uses
    partial(super().__call__, **kwargs) which pre-binds keyword args; if the
    wrapper also unpacked 'hidden_states' from *args and then re-injected it,
    position_embeddings appeared in both args and kwargs → TypeError.

    Fix: assign a plain function (no MethodType). Instance-level function
    attributes in Python are NOT subject to descriptor binding, so __call__
    calls _fwd(*args, **kwargs) exactly as given with no implicit self
    prepended. We extract hidden_states from args[0] ONLY for the skip path
    and otherwise pass *args/**kwargs through untouched.
    """
    layers = find_transformer_layers(model)
    if layers is None:
        print("  Stochastic Depth: could not find layers, skipping.")
        return model

    n         = len(layers)
    start_idx = int(n * start_fraction)
    patched   = 0

    for i, layer in enumerate(layers):
        if i < start_idx:
            continue

        drop_p = max_dropout * ((i - start_idx) / max(n - start_idx - 1, 1))
        orig   = layer.forward          # bound method -- self already captured

        def _make(orig_fn, dp):
            def _fwd(*args, **kwargs):
                # args[0] is hidden_states regardless of whether we are inside
                # gradient checkpointing or a normal forward call.
                hidden_states = args[0]

                if not torch.is_grad_enabled():   # inference / GC recompute: never skip
                    return orig_fn(*args, **kwargs)

                if torch.rand(1).item() < dp:     # training: probabilistic skip
                    out = (hidden_states,)
                    if kwargs.get("output_attentions", False): out += (None,)
                    if kwargs.get("use_cache",         False): out += (None,)
                    return out

                return orig_fn(*args, **kwargs)   # normal forward
            return _fwd

        # Assign as plain instance attribute -- NO types.MethodType.
        # Python does not apply the descriptor protocol to instance attributes,
        # so nn.Module.__call__ will invoke _fwd(*args, **kwargs) directly.
        layer.forward = _make(orig, drop_p)
        patched += 1

    print(f"  Stochastic Depth: {patched}/{n} layers wrapped "
          f"(first {start_idx} protected, max_p={max_dropout})")
    return model


def verify_mor(model):
    # FIX v2: LoraLayer moved to peft.tuners.lora in newer PEFT versions.
    try:
        from peft.tuners.lora import LoraLayer
    except ImportError:
        try:
            from peft import LoraLayer
        except ImportError:
            print("  MoR verify: LoraLayer not found in PEFT -- skipping verify.")
            return

    mlp_names  = ["gate_proj", "up_proj", "down_proj"]
    attn_names = ["q_proj", "k_proj", "v_proj", "o_proj"]
    mlp_r, attn_r = [], []

    for name, mod in model.named_modules():
        if not isinstance(mod, LoraLayer) or not hasattr(mod, "r"):
            continue
        rv = list(mod.r.values())[0] if isinstance(mod.r, dict) else mod.r
        if   any(x in name for x in mlp_names):  mlp_r.append(rv)
        elif any(x in name for x in attn_names): attn_r.append(rv)

    if mlp_r and attn_r:
        print(f"  MoR verify: attn r={set(attn_r)}, mlp r={set(mlp_r)}")
        if set(mlp_r) == set(attn_r):
            print("  MoR: rank_pattern ignored -- applying manual MLP fallback")
            for name, mod in model.named_modules():
                if isinstance(mod, LoraLayer) and any(x in name for x in mlp_names):
                    if isinstance(mod.r, dict):
                        mod.r          = {k: CFG["lora_r_mlp"]     for k in mod.r}
                        mod.lora_alpha = {k: CFG["lora_alpha_mlp"] for k in mod.lora_alpha}
                    else:
                        mod.r          = CFG["lora_r_mlp"]
                        mod.lora_alpha = CFG["lora_alpha_mlp"]
            print(f"  MoR: MLP fallback applied (r={CFG['lora_r_mlp']}, α={CFG['lora_alpha_mlp']})")
        else:
            print("  MoR: rank_pattern working correctly ✓")
    else:
        print("  MoR: no LoraLayer modules found to verify")

# =============================================================================
#  EMA (Exponential Moving Average) -- shadow params for stable inference
# =============================================================================

from transformers import TrainerCallback as _TrainerCallback

class EMACallback(_TrainerCallback):
    """Maintains an EMA shadow of trainable parameters.
    After training, swap shadow in for more stable generation."""
    def __init__(self, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self._initialized = False

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None: return
        if not self._initialized:
            for n, p in model.named_parameters():
                if p.requires_grad:
                    self.shadow[n] = p.data.clone()
            self._initialized = True
            return
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def swap_in(self, model):
        """Replace model params with EMA shadow (call before generation)."""
        self._backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self._backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def swap_out(self, model):
        """Restore original params after generation."""
        for n, p in model.named_parameters():
            if n in self._backup:
                p.data.copy_(self._backup[n])
        self._backup = {}

# =============================================================================
#  STAGE 2 -- SFT
# =============================================================================

def stage2_finetune():
    banner("STAGE 2 -- SFT: QDoRA + rsLoRA + NEFTune + MoR + Stochastic Depth + EMA + Curriculum")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    os.environ["PYTORCH_ALLOC_CONF"]      = "expandable_segments:True,max_split_size_mb:128"
    os.environ["TOKENIZERS_PARALLELISM"]  = "false"
    os.environ["HF_HUB_OFFLINE"]          = "1"
    os.environ["TRANSFORMERS_OFFLINE"]    = "1"

    if os.path.exists(os.path.join(CFG["output_dir"], "adapter_config.json")):
        print("  SFT adapters already exist. Skipping.")
        set_stage(3); return

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    with open(DATA_FILE) as f:
        base_count = sum(1 for _ in f)
    print(f"  Base dataset: {base_count:,} examples")

    print("  Sampling sequence lengths (first 5k)...")
    lengths = []
    with open(DATA_FILE) as f:
        for i, line in enumerate(f):
            if i >= 5000: break
            ex    = json.loads(line)
            chars = sum(len(c.get("content","")) for c in ex.get("conversations",[]))
            lengths.append(chars)
    lengths.sort()
    p95 = lengths[int(len(lengths)*0.95)] // 4
    print(f"  p95 token estimate: {p95} "
          f"({'fits' if p95 < CFG['max_seq_length'] else 'consider increasing max_seq_length'})")

    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template, standardize_data_formats
    from datasets import load_dataset as hf_load, Dataset, concatenate_datasets
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainerCallback
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from IPython.display import display

    class LiveLossPlot(TrainerCallback):
        def __init__(self):
            self.steps = []; self.losses = []
            self.handle = self.fig = self.ax = None
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "loss" not in logs: return
            if self.handle is None:
                self.fig, self.ax = plt.subplots(figsize=(10, 4))
                self.handle = display(self.fig, display_id=True)
            self.steps.append(state.global_step)
            self.losses.append(logs["loss"])
            self.ax.clear()
            self.ax.plot(self.steps, self.losses, color="#1f77b4", lw=2.5,
                         marker="o", ms=4, mfc="#ff7f0e")
            bi = self.losses.index(min(self.losses))
            self.ax.axhline(self.losses[bi], color="#2ca02c", ls="--", lw=1.2,
                            alpha=0.7, label=f"Best {self.losses[bi]:.4f} @ {self.steps[bi]}")
            self.ax.legend(fontsize=9)
            self.ax.set_title(
                f"SchoolAI SFT  [step {state.global_step} | loss {logs['loss']:.4f}]",
                fontsize=13, fontweight="bold")
            self.ax.set_xlabel("Step"); self.ax.set_ylabel("Loss")
            self.ax.grid(True, ls="--", alpha=0.5); self.fig.tight_layout()
            self.handle.update(self.fig)

    num_gpus   = torch.cuda.device_count()
    # BNB 4-bit cannot split across GPUs for training. Everything on GPU 0.
    for i in range(num_gpus):
        name = torch.cuda.get_device_name(i)
        mem  = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {name} ({mem:.1f} GB)")
    try:
        import transformers.modeling_utils as _tmu
        _tmu.caching_allocator_warmup = lambda *a, **kw: None
        print("  caching_allocator_warmup disabled.")
    except: pass
    print("\n  Loading base model...")
    model, tokenizer = FastModel.from_pretrained(
        model_name      = CFG["model"],
        dtype           = None,
        max_seq_length  = CFG["max_seq_length"],
        load_in_4bit    = True,
        full_finetuning = False,
        device_map      = CFG["device_map"],
    )
    vram_total = sum(
        torch.cuda.memory_reserved(i) / 1024**3 for i in range(num_gpus)
    )
    vram_per   = [f"GPU{i}: {torch.cuda.memory_reserved(i)/1024**3:.1f}GB"
                  for i in range(num_gpus)]
    print(f"  Loaded. VRAM total: {vram_total:.1f} GB  ({', '.join(vram_per)})")

    # FIX v2: Silence the gradient-checkpointing/cache warning by setting
    # use_cache=False on the config before any forward pass.
    model.config.use_cache = False

    if CFG["sd_enabled"]:
        model = apply_stochastic_depth(
            model, CFG["sd_max_dropout"], CFG["sd_start_fraction"]
        )

    print(f"\n  Attaching QDoRA + rsLoRA + MoR "
          f"(attn r={CFG['lora_r']} α={CFG['lora_alpha']}, "
          f"mlp r={CFG['lora_r_mlp']} α={CFG['lora_alpha_mlp']})...")
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers     = False,
        finetune_language_layers   = True,
        finetune_attention_modules = True,
        finetune_mlp_modules       = True,
        r            = CFG["lora_r"],
        lora_alpha   = CFG["lora_alpha"],
        lora_dropout = 0,
        bias         = "none",
        use_dora     = CFG["use_dora"],
        use_rslora   = CFG["use_rslora"],
        random_state = 3407,
        # With lora_r == lora_r_mlp == 8, rank_pattern is uniform.
        # Kept explicit so MoR can be re-enabled later with different ranks.
        rank_pattern = {
            "q_proj": CFG["lora_r"],     "k_proj": CFG["lora_r"],
            "v_proj": CFG["lora_r"],     "o_proj": CFG["lora_r"],
            "gate_proj": CFG["lora_r_mlp"],
            "up_proj":   CFG["lora_r_mlp"],
            "down_proj": CFG["lora_r_mlp"],
        },
        alpha_pattern = {
            "q_proj": CFG["lora_alpha"],     "k_proj": CFG["lora_alpha"],
            "v_proj": CFG["lora_alpha"],     "o_proj": CFG["lora_alpha"],
            "gate_proj": CFG["lora_alpha_mlp"],
            "up_proj":   CFG["lora_alpha_mlp"],
            "down_proj": CFG["lora_alpha_mlp"],
        },
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
    verify_mor(model)

    # Single GPU: no need for model_parallel flags

    # go online for HF dataset fetching
    os.environ.pop("HF_HUB_OFFLINE",      None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    tokenizer    = get_chat_template(tokenizer, chat_template="gemma-4")
    base_dataset = hf_load("json", data_files=DATA_FILE, split="train")
    base_dataset = standardize_data_formats(base_dataset)
    print(f"\n  Base: {len(base_dataset):,}")

    # ── 5% replay: curated domain anchors ──────────────────────────────
    anchor_rows = [make_convo(SUBJECTS.get(s, HONESTY_SYSTEM), q, a)
                   for s, q, a in CURATED_ANCHORS]
    anchor_ds   = standardize_data_formats(Dataset.from_list(anchor_rows))
    replay_n    = max(1, int(base_count * CFG["replay_fraction"]) // len(anchor_rows))
    anchor_ds   = concatenate_datasets([anchor_ds] * replay_n)
    print(f"  Domain replay: {len(anchor_ds):,} "
          f"({CFG['replay_fraction']*100:.0f}% target, {replay_n}x each anchor)")

    # ── 10% chat: external prose + CHAT_ANCHORS voice ──────────────────
    chat_target = int(base_count * CFG["chat_anchor_fraction"])  # ~10,000

    ext_rows = fetch_external_chat_datasets()
    ext_ds   = standardize_data_formats(Dataset.from_list(ext_rows)) if ext_rows else None
    print(f"  External chat: {len(ext_rows):,}")

    chat_rows      = [make_convo(sys, u, a) for sys, u, a in CHAT_ANCHORS]
    chat_single_ds = standardize_data_formats(Dataset.from_list(chat_rows))

    # FIX v2: if ext datasets all failed, the voice anchors must fill the
    # entire chat_target slot. Recalculate remaining accordingly.
    ext_count  = len(ext_rows)
    remaining  = max(1, (chat_target - ext_count) // len(chat_rows))
    voice_ds   = concatenate_datasets([chat_single_ds] * remaining)
    print(f"  SchoolAI voice anchors: {len(voice_ds):,} "
          f"({remaining}x each, fills remaining chat slot)")

    chat_combined = concatenate_datasets([ext_ds, voice_ds] if ext_ds else [voice_ds])
    if len(chat_combined) > chat_target:
        chat_combined = chat_combined.shuffle(seed=3407).select(range(chat_target))
    print(f"  Total chat slot: {len(chat_combined):,} / target {chat_target:,}")

    dataset = concatenate_datasets([base_dataset, anchor_ds, chat_combined])
    dataset = dataset.shuffle(seed=3407)
    print(f"  Grand total: {len(dataset):,}")

    def apply_template(examples):
        texts = []
        for convo in examples["conversations"]:
            text = tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            if not text.endswith(tokenizer.eos_token):
                text += tokenizer.eos_token
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(apply_template, batched=True, desc="Formatting")

    # ── Token-level curriculum: sort easy→hard by text length ────────────
    if CFG["curriculum_enabled"]:
        def _text_len(example):
            return {"_sort_len": len(example.get("text", ""))}
        dataset = dataset.map(_text_len, desc="Computing curriculum lengths")
        dataset = dataset.sort("_sort_len")
        dataset = dataset.remove_columns(["_sort_len"])
        print(f"  Curriculum: sorted {len(dataset):,} examples easy→hard by length")

    print(f"  Sample:\n{dataset[0]['text'][:300]}\n")

    # FIX v2: free dataset-build memory before the trainer allocates
    gc.collect()
    torch.cuda.empty_cache()

    eff_batch = CFG["batch_size"] * CFG["grad_accum"] * num_gpus
    print(f"  Effective batch: {eff_batch}")

    ema_cb = EMACallback(decay=CFG["ema_decay"]) if CFG["ema_enabled"] else None
    callbacks = [LiveLossPlot()]
    if ema_cb: callbacks.append(ema_cb)

    trainer = SFTTrainer(
        model         = model,
        tokenizer     = tokenizer,
        train_dataset = dataset,
        callbacks     = callbacks,
        args          = SFTConfig(
            dataset_text_field          = "text",
            per_device_train_batch_size = CFG["batch_size"],
            gradient_accumulation_steps = CFG["grad_accum"],
            num_train_epochs            = CFG["epochs"],
            max_steps                   = CFG["max_steps"] if CFG["max_steps"] > 0 else -1,
            warmup_steps                = CFG["warmup_steps"],
            learning_rate               = CFG["lr"],
            lr_scheduler_type           = "cosine",
            optim                       = "adamw_8bit",
            weight_decay                = 0.01,
            max_grad_norm               = 1.0,
            fp16                        = not torch.cuda.is_bf16_supported(),
            bf16                        = torch.cuda.is_bf16_supported(),
            fp16_full_eval              = False,   # prevents _move_model_to_device on multi-GPU
            bf16_full_eval              = False,
            neftune_noise_alpha         = CFG["neftune_noise_alpha"],
            packing                     = True,
            max_seq_length              = CFG["max_seq_length"],
            logging_steps               = 25,
            save_strategy               = "steps",
            save_steps                  = 100,
            save_total_limit            = 5,
            output_dir                  = CFG["output_dir"],
            report_to                   = "none",
            seed                        = 3407,
            gradient_checkpointing      = "unsloth",   # 30% less VRAM than standard
            gradient_checkpointing_kwargs = {"use_reentrant": False},
            remove_unused_columns       = False,
            dataloader_num_workers      = 0,
            dataloader_pin_memory       = False,
        ),
    )


    import glob
    # Resume from a specific checkpoint, or auto-detect the latest one
    explicit_ckpt = ""
    checkpoints = glob.glob(os.path.join(CFG["output_dir"], "checkpoint-*"))
    
    if explicit_ckpt and os.path.exists(explicit_ckpt):
        resume = explicit_ckpt
        print(f"\n  Resuming from explicit checkpoint: {resume}")
    elif checkpoints:
        resume = True  # auto-detect latest
        print(f"\n  Found {len(checkpoints)} checkpoint(s). Resuming from latest...")
    else:
        resume = None
        print("\n  No checkpoints found. Starting fresh.")

    result   = trainer.train(resume_from_checkpoint=resume)
    runtime  = result.metrics["train_runtime"]
    loss     = result.metrics.get("train_loss", "N/A")
    mem_peak = torch.cuda.max_memory_reserved() / 1024**3
    print(f"\n  Done. {runtime/60:.1f} min | loss={loss} | peak VRAM={mem_peak:.1f} GB")

    os.makedirs(CFG["output_dir"], exist_ok=True)
    model.save_pretrained(CFG["output_dir"])
    tokenizer.save_pretrained(CFG["output_dir"])

    with open(os.path.join(CFG["output_dir"], "training_meta.json"), "w") as f:
        json.dump({
            "base": base_count, "replay": len(anchor_ds),
            "chat_ext": ext_count, "chat_voice": len(voice_ds),
            "total": len(dataset),
            "techniques": "QDoRA+rsLoRA+MoR+NEFTune+StochasticDepth+EMA+Curriculum",
            "lora_r_attn": CFG["lora_r"], "lora_alpha_attn": CFG["lora_alpha"],
            "lora_r_mlp":  CFG["lora_r_mlp"], "lora_alpha_mlp": CFG["lora_alpha_mlp"],
            "sd_max_dropout": CFG["sd_max_dropout"],
            "neftune": CFG["neftune_noise_alpha"],
            "ema_enabled": CFG["ema_enabled"],
            "runtime_s": runtime, "loss": str(loss), "peak_vram_gb": mem_peak,
        }, f, indent=2)

    # Save EMA shadow weights to disk for stage4 eval
    if ema_cb and ema_cb._initialized:
        import safetensors
        ema_path = os.path.join(CFG["output_dir"], "ema_shadow.safetensors")
        ema_tensors = {k.replace(".", "-"): v for k, v in ema_cb.shadow.items()}
        safetensors.torch.save_file(ema_tensors, ema_path)
        print(f"  EMA shadow saved → {ema_path}")

    global _model, _tokenizer
    _model, _tokenizer = model, tokenizer
    set_stage(3)

# =============================================================================
#  STAGE 3 -- GRPO  (tutoring behavior alignment)
# =============================================================================

GRPO_PROMPTS_EN = [
    [{"role": "user", "content": "Make me a game"}],
    [{"role": "user", "content": "Build me a website"}],
    [{"role": "user", "content": "Help me with my homework"}],
    [{"role": "user", "content": "Write something about history"}],
    [{"role": "user", "content": "Explain everything about physics"}],
    [{"role": "user", "content": "Solve step by step: 2x + 5 = 13"}],
    [{"role": "user", "content": "Explain photosynthesis step by step"}],
    [{"role": "user", "content": "Write a Python function to reverse a string"}],
    [{"role": "user", "content": "What is the exact population of Albania right now?"}],
    [{"role": "user", "content": "What will the economy do next year?"}],
    [{"role": "user", "content": "I got x=8 for 3x-1=20. Is that right?"}],
    [{"role": "user", "content": "I got 42 but the answer key says 56"}],
    [{"role": "user", "content": "I'm really stressed about my exam tomorrow"}],
    [{"role": "user", "content": "I give up, this is too hard"}],
    [{"role": "user", "content": "I hate studying"}],
    [{"role": "user", "content": "Write my essay for me"}],
    [{"role": "user", "content": "Can I submit your answer as my homework?"}],
    [{"role": "user", "content": "I got 95 on my test!"}],
    [{"role": "user", "content": "You explained that really well, thanks"}],
]

GRPO_PROMPTS_ML = [
    # ── German ────────────────────────────────────────────────────────────
    [{"role": "user", "content": "Erkläre mir den Unterschied zwischen Akkusativ und Dativ."}],
    [{"role": "user", "content": "Wie schreibt man eine Erörterung?"}],
    [{"role": "user", "content": "Ich verstehe den Konjunktiv II nicht."}],
    [{"role": "user", "content": "Hilf mir bei meinen Hausaufgaben"}],
    [{"role": "user", "content": "Was ist der Unterschied zwischen weil und denn?"}],
    # ── Spanish ────────────────────────────────────────────────────────────
    [{"role": "user", "content": "¿Cuándo se usa el pretérito y cuándo el imperfecto?"}],
    [{"role": "user", "content": "Explícame la diferencia entre ser y estar."}],
    [{"role": "user", "content": "No entiendo el subjuntivo."}],
    [{"role": "user", "content": "Ayúdame con mi tarea"}],
    [{"role": "user", "content": "¿Cómo se escriben los acentos en español?"}],
    # ── French ─────────────────────────────────────────────────────────────
    [{"role": "user", "content": "Quelle est la différence entre l'imparfait et le passé composé?"}],
    [{"role": "user", "content": "Comment utiliser les pronoms relatifs?"}],
    [{"role": "user", "content": "Je ne comprends pas le subjonctif."}],
    [{"role": "user", "content": "Aidez-moi avec mes devoirs"}],
    [{"role": "user", "content": "Comment rédiger une dissertation?"}],
    # ── Italian ────────────────────────────────────────────────────────────
    [{"role": "user", "content": "Qual è la differenza tra imperfetto e passato prossimo?"}],
    [{"role": "user", "content": "Come si usano gli articoli determinativi?"}],
    [{"role": "user", "content": "Non capisco il congiuntivo."}],
    [{"role": "user", "content": "Aiutami con i compiti"}],
    [{"role": "user", "content": "Come si scrive un tema?"}],
    # ── Albanian ────────────────────────────────────────────────────────────
    [{"role": "user", "content": "Si formohen kohet e foljes ne shqip?"}],
    [{"role": "user", "content": "Shpjego figurat e stilit ne gjuhen shqipe."}],
    [{"role": "user", "content": "Nuk e kuptoj temen. Ku te filloj?"}],
    [{"role": "user", "content": "Me ndihmo me detyrat e shtepise"}],
    [{"role": "user", "content": "Si shkruhet nje ese analitike ne shqip?"}],
]

GRPO_PROMPTS = (GRPO_PROMPTS_EN * 20 + GRPO_PROMPTS_ML * 10)  # ~380 EN + ~250 ML = ~630

def _text(c):
    return c[0]["content"] if isinstance(c, list) else c

def reward_structured_steps(completions, **kwargs):
    rewards = []
    for c in completions:
        t     = _text(c)
        score = 0.0
        if re.search(r"\*\*\d+[\.\)]|\n\d+[\.\)]", t):               score += 0.4
        if re.search(r"=\s*[\d\-]|→|\btherefore\b|\bthus\b", t, re.I): score += 0.3
        if re.search(r"\bcheck\b|\bverif|\bplug in\b|\bconfirm\b", t, re.I): score += 0.3
        rewards.append(score)
    return rewards

def reward_clarifies_vague(completions, prompts=None, **kwargs):
    vague = ["make me","build me","build a","create a","write something",
             "help me with my homework","explain everything"]
    rewards = []
    for i, c in enumerate(completions):
        t  = _text(c)
        pt = ""
        if prompts:
            p  = prompts[i]
            pt = (p[-1]["content"] if isinstance(p, list) else p).lower()
        is_vague = any(s in pt for s in vague)
        has_q    = "?" in t
        has_list = bool(re.search(r"\n\d+\.", t))
        if   is_vague and has_q and has_list: rewards.append(1.0)
        elif is_vague and has_q:              rewards.append(0.6)
        elif is_vague and not has_q:          rewards.append(-0.5)
        else:                                 rewards.append(0.2)
    return rewards

def reward_honest_uncertainty(completions, prompts=None, **kwargs):
    uncertain_signals = ["exact","right now","current","population","today",
                         "weather","price","next year","latest"]
    hedge_phrases     = ["i'm not sure","i don't know","i can't confirm",
                         "you should verify","check with","my knowledge",
                         "i don't have access","may have changed","i'd recommend"]
    overconfident     = [r"\bexactly\b", r"\bprecisely\b", r"\b100%\b"]
    rewards = []
    for i, c in enumerate(completions):
        t  = _text(c)
        pt = ""
        if prompts:
            p  = prompts[i]
            pt = (p[-1]["content"] if isinstance(p, list) else p).lower()
        needs_hedge = any(s in pt for s in uncertain_signals)
        has_hedge   = any(p in t.lower() for p in hedge_phrases)
        overconf    = any(bool(re.search(p, t, re.I)) for p in overconfident)
        if   needs_hedge and has_hedge and not overconf: rewards.append(1.0)
        elif needs_hedge and has_hedge:                  rewards.append(0.5)
        elif needs_hedge and not has_hedge:              rewards.append(-0.3)
        else:                                            rewards.append(0.2)
    return rewards

def reward_natural_register(completions, **kwargs):
    robotic = ["certainly!","of course!","absolutely!","great question!",
               "as an ai","i am an ai","i'm an ai","i'd be happy to assist",
               "i'd be delighted","i'd be glad to"]
    natural = ["let's","let me","sure","here's","i can","of course,",
               "happy to","good","nice","that feeling","that happens"]
    rewards = []
    for c in completions:
        t     = _text(c).lower()
        first = t[:100]
        score = 0.0
        if any(p in first for p in robotic): score -= 0.5
        if any(p in first for p in natural): score += 0.3
        rewards.append(score)
    return rewards

def reward_length(completions, **kwargs):
    rewards = []
    for c in completions:
        words = len(_text(c).split())
        if   words < 15:  rewards.append(-0.6)
        elif words < 40:  rewards.append(0.1)
        elif words < 300: rewards.append(0.5)
        elif words < 500: rewards.append(0.2)
        else:             rewards.append(-0.2)
    return rewards

def reward_language_match(completions, prompts=None, **kwargs):
    """Penalize responding in the wrong language.
    If the prompt is in German/Spanish/French/Italian/Albanian,
    the response should contain words from that language."""
    # Common function words that strongly signal the language
    lang_markers = {
        "de": ["ich","du","er","sie","wir","ist","sind","hat","haben","und","oder",
               "aber","nicht","ein","eine","der","die","das","den","dem","des",
               "mit","für","auf","aus","bei","nach","seit","von","zu","kann",
               "werden","wurde","wäre","hätte","könnte","müsste","sollte","dass"],
        "es": ["yo","tú","él","ella","nosotros","es","son","está","están","y","o",
               "pero","no","un","una","el","la","los","las","en","de","por","para",
               "con","sin","sobre","entre","puede","puedo","quiero","necesito",
               "porque","aunque","cuando","donde","como","qué","cómo","cuál"],
        "fr": ["je","tu","il","elle","nous","vous","ils","est","sont","a","ont",
               "et","ou","mais","ne","pas","un","une","le","la","les","en","de",
               "du","des","pour","avec","dans","sur","par","peut","puis","veux",
               "doit","fait","avoir","être","que","qui","dont","où","comment"],
        "it": ["io","tu","lui","lei","noi","voi","loro","è","sono","ha","hanno",
               "e","ma","non","un","uno","una","il","lo","la","i","gli","le",
               "in","di","per","con","su","tra","può","posso","voglio","devo",
               "che","perché","quando","dove","come","quale","anche","ancora"],
        "sq": ["une","ti","ai","ajo","ne","ju","ata","eshte","jane","ka","kane",
               "dhe","apo","por","nuk","nje","i","e","te","ne","me","per","nga",
               "me","pa","si","ose","mund","duhet","do","kam","jemi","janë",
               "eshtë","shqip","shqipe","shqiperise","gjuhen","gjuha"],
    }
    # Detect prompt language by counting marker hits (word-boundary matching)
    prompt_lang_markers = {
        "de": [r"\bich\b",r"\bdu\b",r"\bwir\b",r"\bist\b",r"\bsind\b",r"\bder\b",r"\bdie\b",r"\bdas\b",r"\bund\b",r"\bnicht\b",
               r"\berkläre\b",r"\bhilfe\b",r"\bhausaufgaben\b",r"\bverstehen\b",r"\bkann\b",r"\bwerden\b",r"\bwäre\b"],
        "es": [r"\byo\b",r"\btú\b",r"\bnosotros\b",r"\bestá\b",r"\bson\b",r"\bel\b",r"\bla\b",r"\blos\b",r"\by\b",r"\bno\b",
               r"\bexplícame\b",r"\bayuda\b",r"\btarea\b",r"\bentiendo\b",r"\bpuede\b",r"\bquiero\b",r"\bcómo\b"],
        "fr": [r"\bje\b",r"\btu\b",r"\bnous\b",r"\best\b",r"\bsont\b",r"\ble\b",r"\bla\b",r"\bles\b",r"\bet\b",r"\bne\b",
               r"\bexpliquez\b",r"\baidez\b",r"\bdevoirs\b",r"\bcomprends\b",r"\bpeut\b",r"\bveux\b",r"\bcomment\b"],
        "it": [r"\bio\b",r"\btu\b",r"\bnoi\b",r"\bè\b",r"\bsono\b",r"\bil\b",r"\blo\b",r"\bla\b",r"\be\b",r"\bnon\b",
               r"\bspiega\b",r"\baiuto\b",r"\bcompiti\b",r"\bcapisco\b",r"\bpuò\b",r"\bvoglio\b",r"\bcome\b"],
        "sq": [r"\bune\b",r"\bti\b",r"\bne\b",r"\beshte\b",r"\bjane\b",r"\bdhe\b",r"\bnuk\b",r"\bnje\b",r"\bper\b",r"\bka\b",
               r"\bshpjego\b",r"\bndihmo\b",r"\bdetyrat\b",r"\bkuptoj\b",r"\bmund\b",r"\bduhet\b",r"\bsi\b"],
    }
    rewards = []
    for i, c in enumerate(completions):
        t  = _text(c).lower()
        pt = ""
        if prompts:
            p  = prompts[i]
            pt = (p[-1]["content"] if isinstance(p, list) else p).lower()
        # Detect prompt language
        best_lang = "en"
        best_count = 0
        for lang, patterns in prompt_lang_markers.items():
            count = sum(1 for pat in patterns if re.search(pat, pt, re.I))
            if count > best_count:
                best_count = count
                best_lang = lang
        # If prompt is non-English, check response contains that language
        if best_lang != "en" and best_count >= 2:
            resp_markers = lang_markers.get(best_lang, [])
            resp_count = sum(1 for m in resp_markers if re.search(rf"\b{re.escape(m)}\b", t, re.I))
            if resp_count >= 3:
                rewards.append(0.8)   # good: responding in matching language
            elif resp_count >= 1:
                rewards.append(0.2)   # partial: some target language words
            else:
                rewards.append(-0.8)  # bad: responding in English to non-English prompt
        else:
            rewards.append(0.0)       # English prompt: no language constraint
    return rewards

def stage3_grpo():
    banner("STAGE 3 -- GRPO: tutoring behavior alignment")

    global _model, _tokenizer

    # 1. Unsloth MUST be imported before transformers/trl to apply speedups
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template
    from datasets import Dataset

    # 2. Safely bypass llm_blender so trl doesn't crash
    import sys
    try:
        import importlib
        _trl_utils = importlib.import_module("trl.import_utils")
        _trl_utils._llm_blender_available = False
    except Exception:
        pass

    if os.path.exists(os.path.join(CFG["grpo_output_dir"], "adapter_config.json")):
        print("  GRPO adapters already exist. Skipping.")
        set_stage(4); return

    try:
        from trl import GRPOTrainer, GRPOConfig
    except ImportError:
        print("  GRPOTrainer not available in this trl version. Skipping GRPO.")
        set_stage(4); return

    if _model is None:
        print("  Loading SFT model...")
        _model, _tokenizer = FastModel.from_pretrained(
            model_name     = CFG["output_dir"],
            max_seq_length = 256,          # shorter for GRPO (saves VRAM)
            load_in_4bit   = True,
        )
        _tokenizer = get_chat_template(_tokenizer, chat_template="gemma-4")

    _model.config.use_cache = False

    # The SFT model loaded from CFG["output_dir"] already has LoRA adapters.
    # Unsloth refuses to call get_peft_model() twice. Just unlock the existing
    # adapter weights so GRPO can continue training them.
    for name, param in _model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
    trainable = sum(p.numel() for p in _model.parameters() if p.requires_grad)
    print(f"  GRPO trainable params: {trainable:,}")

    # ── Reference model ───────────────────────────────────────────────
    # Newer trl automatically uses the base model under the LoRA adapters
    # as the reference (disables adapters temporarily for KL computation).
    # No need to load a separate ref model — saves memory and time.
    print("  Using LoRA base model as implicit reference (no separate load needed).")

    # GRPOTrainer expects model.warnings_issued dict (from PreTrainedModel)
    # but PEFT-wrapped Gemma4 doesn't expose it. Add it manually.
    if not hasattr(_model, 'warnings_issued'):
        _model.warnings_issued = {}

    # Clear VRAM before trainer init
    gc.collect()
    torch.cuda.empty_cache()

    trainer = GRPOTrainer(
        model            = _model,
        processing_class = _tokenizer,
        reward_funcs  = [
            reward_structured_steps,
            reward_clarifies_vague,
            reward_honest_uncertainty,
            reward_natural_register,
            reward_length,
            reward_language_match,
        ],
        args = GRPOConfig(
            num_generations             = CFG["grpo_num_generations"],
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 16,
            num_train_epochs            = CFG["grpo_epochs"],
            learning_rate               = CFG["grpo_lr"],
            fp16                        = True,
            bf16                        = False,
            output_dir                  = CFG["grpo_output_dir"],
            report_to                   = "none",
            logging_steps               = 10,
            max_completion_length       = 128,
            max_prompt_length           = 128,
            gradient_checkpointing      = True,
            gradient_checkpointing_kwargs = {"use_reentrant": False},
            seed                        = 3407,
        ),
        train_dataset = Dataset.from_dict({"prompt": GRPO_PROMPTS}),
    )
    trainer.train()
    _model.save_pretrained(CFG["grpo_output_dir"])
    _tokenizer.save_pretrained(CFG["grpo_output_dir"])

    gc.collect()

    set_stage(4)
    print(f"  GRPO complete → {CFG['grpo_output_dir']}")

# =============================================================================
#  STAGE 4 -- GGUF Export + Evaluation Report
# =============================================================================

EVAL_QUESTIONS = [
    (None,       "Hey! Who are you?"),
    (None,       "I'm really stressed about my exam tomorrow."),
    (None,       "I give up. This is too hard."),
    (None,       "Make me a game"),
    (None,       "Can I submit your answer as my homework?"),
    (None,       "What is the exact population of Albania right now?"),
    ("albanian", "Shpjego figurat e stilit ne gjuhen shqipe."),
    ("albanian", "Si formohen kohet e foljes ne shqip?"),
    ("german",   "Erkläre mir den Unterschied zwischen Akkusativ und Dativ."),
    ("spanish",  "¿Cuándo se usa el pretérito y cuándo el imperfecto?"),
    ("french",   "Quelle est la différence entre l'imparfait et le passé composé?"),
    ("italian",  "Qual è la differenza tra imperfetto e passato prossimo?"),
    ("math",     "Solve step by step: A train travels 120 km in 1.5 hours. What is its speed in m/s?"),
    ("physics",  "Explain Newton's second law with a real-world example."),
    ("coding",   "Write a Python function that checks if a string is a palindrome."),
]

def stage4_export():
    banner("STAGE 4 -- GGUF Export + Evaluation Report")

    global _model, _tokenizer

    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template

    best_dir = (CFG["grpo_output_dir"]
                if os.path.exists(os.path.join(CFG["grpo_output_dir"], "adapter_config.json"))
                else CFG["output_dir"])
    gguf_dir    = best_dir + "_gguf"
    report_path = os.path.join(WORKDIR, "evaluation_report.md")

    if _model is None:
        print(f"  Loading model from {best_dir}...")
        _model, _tokenizer = FastModel.from_pretrained(
            model_name     = best_dir,
            max_seq_length = CFG["max_seq_length"],
            load_in_4bit   = True,
        )
        _tokenizer = get_chat_template(_tokenizer, chat_template="gemma-4")

    def gen(model, tok, subject, question, max_new=300):
        sys   = SCHOOLAI_SYSTEM if subject is None else SUBJECTS.get(subject, HONESTY_SYSTEM)
        msgs  = [{"role":"system","content":sys}, {"role":"user","content":question}]
        # Gemma 4 is multimodal -- the processor expects content as list-of-dicts.
        # For text-only generation, use the underlying tokenizer directly.
        _tok  = tok.tokenizer if hasattr(tok, 'tokenizer') else tok
        ids   = _tok.apply_chat_template(msgs, tokenize=True,
                    add_generation_prompt=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new, temperature=0.7,
                                 top_p=0.9, do_sample=True,
                                 pad_token_id=_tok.eos_token_id)
        return _tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    print("  Loading base model for comparison...")
    base_m, base_t = FastModel.from_pretrained(
        model_name=CFG["model"], dtype=None, max_seq_length=512, load_in_4bit=True
    )
    base_t = get_chat_template(base_t, chat_template="gemma-4")

    lines = [
        "# SchoolAI Evaluation Report",
        f"\nModel: {best_dir}",
        f"Stack: QDoRA + rsLoRA + MoR + Stochastic Depth + NEFTune + EMA + Curriculum + GRPO",
        "---\n",
    ]

    # Use EMA shadow weights for evaluation if available
    ema_shadow_path = os.path.join(best_dir, "ema_shadow.safetensors")
    ema_backup = {}
    if os.path.exists(ema_shadow_path):
        try:
            import safetensors
            ema_tensors = safetensors.torch.load_file(ema_shadow_path)
            # Restore key format (dots were replaced with dashes)
            ema_shadow = {k.replace("-", "."): v for k, v in ema_tensors.items()}
            print("  Swapping in EMA shadow weights for evaluation...")
            for n, p in _model.named_parameters():
                if n in ema_shadow:
                    ema_backup[n] = p.data.clone()
                    p.data.copy_(ema_shadow[n].to(p.device))
        except Exception as e:
            print(f"  EMA shadow load failed: {e}")
            ema_backup = {}

    for subject, question in EVAL_QUESTIONS:
        label = subject.upper() if subject else "CHAT"
        print(f"  [{label}] {question[:55]}...")
        lines += [
            f"## [{label}] {question}\n",
            f"### Base\n```\n{gen(base_m, base_t, subject, question)}\n```\n",
            f"### SchoolAI\n```\n{gen(_model, _tokenizer, subject, question)}\n```\n",
            "---\n",
        ]

    if ema_backup:
        for n, p in _model.named_parameters():
            if n in ema_backup:
                p.data.copy_(ema_backup[n])
        print("  Restored original weights after evaluation.")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Report → {report_path}")

    del base_m, base_t
    gc.collect(); torch.cuda.empty_cache()

    if not os.path.exists(gguf_dir):
        print(f"\n  Converting to GGUF ({CFG['gguf_quant'].upper()})...")
        _model.save_pretrained_gguf(gguf_dir, _tokenizer,
                                    quantization_method=CFG["gguf_quant"])
        print(f"  GGUF → {gguf_dir}/")
    else:
        print("  GGUF already exists. Skipping.")

    with open(os.path.join(gguf_dir, "Modelfile"), "w") as f:
        f.write(f"""FROM ./unsloth.{CFG['gguf_quant'].upper()}.gguf

SYSTEM \"\"\"You are SchoolAI, a friendly and expert educational AI tutor built on Gemma 4. You help students across Europe and North America with Mathematics, Physics, Chemistry, Biology, History, Economics, English, French, German, Spanish, Italian, Albanian, Computer Science, and Cybersecurity. You respond in the same language the student uses. You are patient, warm, and encouraging. You ask clarifying questions before answering vague requests. You admit uncertainty rather than inventing answers. You help students think for themselves rather than doing the work for them.\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx {CFG['max_seq_length']}
""")
    print(f"  Modelfile → {gguf_dir}/Modelfile")

    if CFG["push_to_hub"]:
        token = CFG["hf_token"] or os.environ.get("HF_TOKEN", "")
        _model.push_to_hub(CFG["push_to_hub"], token=token)
        _tokenizer.push_to_hub(CFG["push_to_hub"], token=token)
        _model.push_to_hub_gguf(f"{CFG['push_to_hub']}-gguf", _tokenizer,
                                 quantization_method=CFG["gguf_quant"], token=token)
        print(f"  https://huggingface.co/{CFG['push_to_hub']}")

    set_stage(5)
    print(f"""
{"=" * 60}
  ALL DONE
{"=" * 60}

  Outputs (Kaggle output tab):
    SFT adapters  → {CFG['output_dir']}/
    GRPO adapters → {CFG['grpo_output_dir']}/
    GGUF model    → {gguf_dir}/
    Eval report   → {report_path}

  Dataset:
    {100000:>10,}  base (your 100k: subjects + tutoring method)
    {5000:>10,}  domain replay (5%: fact anchors)
    {10000:>10,}  chat anchors (10%: voice + tone)
    {115000:>10,}  total

  Deploy:
    cd {gguf_dir}
    ollama create schoolai -f Modelfile
    ollama run schoolai

  GGUF Q4_K_M ≈ 3.5 GB
{"=" * 60}
""")

# =============================================================================
#  MAIN
# =============================================================================

_model, _tokenizer = None, None

stage = get_stage()
print(f"SchoolAI -- stage {stage}")
print(f"QDoRA  rsLoRA  attn r={CFG['lora_r']} α={CFG['lora_alpha']}  "
      f"mlp r={CFG['lora_r_mlp']} α={CFG['lora_alpha_mlp']}  "
      f"SD max_p={CFG['sd_max_dropout']}  NEFTune={CFG['neftune_noise_alpha']}")
print(f"replay={CFG['replay_fraction']*100:.0f}%  "
      f"chat={CFG['chat_anchor_fraction']*100:.0f}%  "
      f"(ext {sum([CFG['ext_ultrachat'],CFG['ext_lima'],CFG['ext_slimorca'],CFG['ext_no_robots']]):,} + voice anchors)")

if   stage == 0: stage0_install()
elif stage == 2: stage2_finetune(); stage3_grpo(); stage4_export()
elif stage == 3: stage3_grpo();     stage4_export()
elif stage == 4: stage4_export()
elif stage == 5: print("  Complete. Delete .schoolai_stage to restart.")
else:            stage0_install()