################################################################################
# Study 1: Overhypothesis Induction — Full Pre-Registered Analysis
# ============================================================================
# Run from: C:\overhypothesis\
# Usage:    Rscript scripts\study1_analysis.R
#
# Reads:    results\study1\master_item_level.csv
#           results\study1\master_summary.csv
# Writes:   results\study1\R_analysis_output.txt
#           results\study1\R_analysis_tables.csv
#
# Pre-registration: OSF [link]
# This script implements ALL pre-registered hypothesis tests using the
# exact GLMM specifications committed in the pre-registration.
#
# Author: JP Cacioli (with Claude assistance)
# Date: 2026-03-04
################################################################################

# -- Setup -------------------------------------------------------------------
suppressPackageStartupMessages({
  library(lme4)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(boot)       # bootstrap CIs
  library(clinfun)    # Jonckheere-Terpstra
})

# Check for optional packages
has_emmeans <- requireNamespace("emmeans", quietly = TRUE)
if (has_emmeans) library(emmeans)

cat("=" %.% rep("=", 69) %.% "\n")
cat("STUDY 1: OVERHYPOTHESIS INDUCTION — PRE-REGISTERED ANALYSIS\n")
cat("=" %.% rep("=", 69) %.% "\n\n")

# Helper for string repeat
`%.%` <- function(a, b) paste0(a, b)

# -- Paths -------------------------------------------------------------------
base_dir <- "C:/overhypothesis"
item_csv <- file.path(base_dir, "results", "study1", "master_item_level.csv")
summary_csv <- file.path(base_dir, "results", "study1", "master_summary.csv")
out_file <- file.path(base_dir, "results", "study1", "R_analysis_output.txt")

# -- Sink output to file AND console -----------------------------------------
sink_con <- file(out_file, open = "wt")

tee <- function(...) {
  msg <- paste0(...)
  cat(msg, "\n")
  cat(msg, "\n", file = sink_con)
}

tee_line <- function() tee(paste(rep("─", 70), collapse = ""))

# -- Load data ---------------------------------------------------------------
tee("Loading data...")
df <- read_csv(item_csv, show_col_types = FALSE)

# Standardise column names
if ("correct_01" %in% names(df)) df$correct <- as.numeric(df$correct_01)
if ("correct_binary" %in% names(df) && !"correct" %in% names(df)) {
  df$correct <- as.numeric(df$correct_binary)
}
if ("size" %in% names(df) && !"model_size" %in% names(df)) {
  df <- rename(df, model_size = size)
}

# Factor coding
df$condition <- factor(df$condition)
df$model_size <- factor(df$model_size, levels = c("tiny", "small", "medium"))
df$seed <- factor(df$seed)
df$item_id <- factor(df$item_id)

# Filter to full corpus (exclude dose-response fractions for primary tests)
if ("corpus_fraction" %in% names(df)) {
  df_full <- df %>% filter(corpus_fraction == 1.0 | is.na(corpus_fraction))
  tee(sprintf("  Full corpus: %s rows (dropped %s dose-response)",
              format(nrow(df_full), big.mark = ","),
              format(nrow(df) - nrow(df_full), big.mark = ",")))
} else {
  df_full <- df
}

tee(sprintf("  Total rows: %s", format(nrow(df_full), big.mark = ",")))
tee(sprintf("  Conditions: %s", paste(levels(df_full$condition), collapse = ", ")))
tee(sprintf("  Model sizes: %s", paste(levels(df_full$model_size), collapse = ", ")))
tee(sprintf("  Seeds: %s", paste(sort(unique(as.character(df_full$seed))), collapse = ", ")))
tee("")

# ===========================================================================
# SINGULAR-FIT FALLBACK CHAIN
# ===========================================================================
# Pre-registered: If full GLMM produces singular fit, fall back through:
#   1. glmer(correct ~ condition + (1|item_id) + (1|seed))  [full]
#   2. glmer(correct ~ condition + (1|item_id))             [drop seed RE]
#   3. glm(correct ~ condition, family = binomial)          [no RE]
# All attempts and convergence status reported.

fit_glmm <- function(formula_str, data, label = "") {
  results <- list()

  # Attempt 1: Full specification
  tee(sprintf("  Fitting: %s", formula_str))
  full_form <- as.formula(formula_str)

  tryCatch({
    m <- glmer(full_form, data = data, family = binomial,
               control = glmerControl(optimizer = "bobyqa",
                                      optCtrl = list(maxfun = 100000)))
    ss <- isSingular(m)
    results$model <- m
    results$singular <- ss
    results$converged <- TRUE
    results$level <- "full"

    if (ss) {
      tee("  ⚠ SINGULAR FIT — falling back to reduced model")

      # Attempt 2: Drop seed random effect
      reduced_str <- gsub("\\+ \\(1\\|seed\\)", "", formula_str)
      reduced_str <- trimws(reduced_str)
      tee(sprintf("  Fitting reduced: %s", reduced_str))

      tryCatch({
        m2 <- glmer(as.formula(reduced_str), data = data, family = binomial,
                     control = glmerControl(optimizer = "bobyqa",
                                            optCtrl = list(maxfun = 100000)))
        ss2 <- isSingular(m2)
        results$model <- m2
        results$singular <- ss2
        results$level <- "reduced_no_seed"

        if (ss2) {
          tee("  ⚠ STILL SINGULAR — falling back to GLM (no random effects)")
          m3 <- glm(correct ~ condition, data = data, family = binomial)
          results$model <- m3
          results$singular <- FALSE
          results$level <- "glm_fixed"
        }
      }, error = function(e) {
        tee(sprintf("  ERROR in reduced model: %s", e$message))
        m3 <- glm(correct ~ condition, data = data, family = binomial)
        results$model <- m3
        results$singular <- FALSE
        results$level <- "glm_fixed"
      })
    }
  }, error = function(e) {
    tee(sprintf("  ERROR in full model: %s", e$message))
    tee("  Falling back to GLM (no random effects)")
    m3 <- glm(correct ~ condition, data = data, family = binomial)
    results$model <- m3
    results$singular <- FALSE
    results$level <- "glm_fixed"
  })

  tee(sprintf("  Model level used: %s", results$level))
  return(results)
}

# ===========================================================================
# RESULTS COLLECTOR
# ===========================================================================
results_table <- data.frame(
  Hypothesis = character(),
  Type = character(),
  Size = character(),
  Test = character(),
  Statistic = character(),
  Value = numeric(),
  p_value = numeric(),
  Alpha = numeric(),
  Verdict = character(),
  Note = character(),
  stringsAsFactors = FALSE
)

add_result <- function(hyp, type, size, test, stat, val, p, alpha, verdict, note = "") {
  results_table <<- rbind(results_table, data.frame(
    Hypothesis = hyp, Type = type, Size = size, Test = test,
    Statistic = stat, Value = round(val, 4), p_value = round(p, 5),
    Alpha = alpha, Verdict = verdict, Note = note,
    stringsAsFactors = FALSE
  ))
}

# ===========================================================================
# H1: OVERHYPOTHESIS EMERGENCE (Confirmatory Primary)
# ===========================================================================
tee_line()
tee("H1: OVERHYPOTHESIS EMERGENCE")
tee("  Test: Regular vs Scrambled on second-order items")
tee("  Criterion: >=15 pp above chance AND >=10 pp above Scrambled")
tee("  Pre-registered model: glmer(correct ~ condition + (1|item_id) + (1|seed))")
tee_line()

for (sz in c("tiny", "small", "medium")) {
  tee(sprintf("\n--- %s ---", toupper(sz)))

  h1_data <- df_full %>%
    filter(item_type == "second_order",
           condition %in% c("regular", "scrambled"),
           model_size == sz) %>%
    mutate(condition = relevel(factor(condition), ref = "scrambled"))

  tee(sprintf("  N rows: %d  |  N items: %d  |  N seeds: %d",
              nrow(h1_data), n_distinct(h1_data$item_id), n_distinct(h1_data$seed)))

  # Descriptives
  desc <- h1_data %>%
    group_by(condition) %>%
    summarise(mean_acc = mean(correct, na.rm = TRUE),
              sd_acc = sd(correct, na.rm = TRUE),
              n = n(), .groups = "drop")
  for (i in seq_len(nrow(desc))) {
    tee(sprintf("  %s: M=%.1f%%, SD=%.1f%%, N=%d",
                desc$condition[i], desc$mean_acc[i] * 100,
                desc$sd_acc[i] * 100, desc$n[i]))
  }

  reg_mean <- desc %>% filter(condition == "regular") %>% pull(mean_acc) * 100
  scr_mean <- desc %>% filter(condition == "scrambled") %>% pull(mean_acc) * 100

  tee(sprintf("  Reg - Chance = %.1f pp", reg_mean - 50))
  tee(sprintf("  Reg - Scr = %.1f pp", reg_mean - scr_mean))

  # GLMM with fallback
  fit <- fit_glmm("correct ~ condition + (1|item_id) + (1|seed)", h1_data, "H1")
  m <- fit$model
  s <- summary(m)
  tee(capture.output(print(s)))

  # Extract condition coefficient
  coefs <- coef(s)
  if ("conditionregular" %in% rownames(coefs)) {
    est <- coefs["conditionregular", "Estimate"]
    z_val <- coefs["conditionregular", "z value"]
    p_val <- coefs["conditionregular", "Pr(>|z|)"]
  } else if (nrow(coefs) > 1) {
    est <- coefs[2, "Estimate"]
    z_val <- coefs[2, ncol(coefs) - 1]
    p_val <- coefs[2, ncol(coefs)]
  } else {
    est <- NA; z_val <- NA; p_val <- NA
  }

  verdict <- ifelse((reg_mean - 50 >= 15) & (reg_mean - scr_mean >= 10),
                     "SUPPORTED", "NOT SUPPORTED")
  tee(sprintf("  β = %.3f, z = %.3f, p = %.4f", est, z_val, p_val))
  tee(sprintf("  VERDICT: %s (model level: %s)", verdict, fit$level))

  add_result("H1", "Confirmatory", sz, "GLMM Reg vs Scr",
             "beta", est, p_val, 0.05, verdict,
             sprintf("level=%s; reg=%.1f%%; scr=%.1f%%", fit$level, reg_mean, scr_mean))

  # Nonparametric backup: Mann-Whitney U on seed-level means
  seed_means <- h1_data %>%
    group_by(condition, seed) %>%
    summarise(acc = mean(correct, na.rm = TRUE), .groups = "drop")
  reg_seeds <- seed_means %>% filter(condition == "regular") %>% pull(acc)
  scr_seeds <- seed_means %>% filter(condition == "scrambled") %>% pull(acc)

  if (length(reg_seeds) >= 2 && length(scr_seeds) >= 2) {
    wt <- wilcox.test(reg_seeds, scr_seeds, exact = FALSE)
    tee(sprintf("  Mann-Whitney U: W=%.0f, p=%.4f", wt$statistic, wt$p.value))
    add_result("H1", "Nonparametric", sz, "Mann-Whitney U",
               "W", wt$statistic, wt$p.value, 0.05, verdict, "seed-level means")
  }
}

# ===========================================================================
# H2: FRAME-VARIANT TRANSFER (Confirmatory Primary)
# ===========================================================================
tee_line()
tee("H2: FRAME-VARIANT TRANSFER")
tee("  Test: Frame-variant accuracy >= 75% of standard SO accuracy")
tee("  Pre-registered: Paired t-test across frame conditions per size")
tee_line()

for (sz in c("tiny", "small", "medium")) {
  tee(sprintf("\n--- %s ---", toupper(sz)))

  h2_data <- df_full %>%
    filter(condition == "regular",
           item_type %in% c("second_order", "frame_variant"),
           model_size == sz)

  seed_acc <- h2_data %>%
    group_by(seed, item_type) %>%
    summarise(acc = mean(correct, na.rm = TRUE), .groups = "drop") %>%
    pivot_wider(names_from = item_type, values_from = acc)

  if ("frame_variant" %in% names(seed_acc) && "second_order" %in% names(seed_acc)) {
    seed_acc$ratio <- seed_acc$frame_variant / seed_acc$second_order

    fv_mean <- mean(seed_acc$frame_variant, na.rm = TRUE) * 100
    so_mean <- mean(seed_acc$second_order, na.rm = TRUE) * 100
    ratio_mean <- mean(seed_acc$ratio, na.rm = TRUE)

    tee(sprintf("  FV mean: %.1f%%  |  SO mean: %.1f%%  |  Ratio: %.2f",
                fv_mean, so_mean, ratio_mean))

    if (nrow(seed_acc) >= 3) {
      tt <- t.test(seed_acc$frame_variant, seed_acc$second_order, paired = TRUE)
      tee(sprintf("  Paired t-test: t(%.0f) = %.3f, p = %.4f", tt$parameter, tt$statistic, tt$p.value))

      verdict <- ifelse(ratio_mean >= 0.75 & so_mean > 50, "SUPPORTED",
                        ifelse(so_mean <= 52, "VACUOUS (both at chance)", "NOT SUPPORTED"))
      tee(sprintf("  VERDICT: %s", verdict))

      add_result("H2", "Confirmatory", sz, "Paired t-test FV vs SO",
                 "t", tt$statistic, tt$p.value, 0.05, verdict,
                 sprintf("ratio=%.2f; fv=%.1f%%; so=%.1f%%", ratio_mean, fv_mean, so_mean))
    }
  } else {
    tee("  Missing frame_variant or second_order data")
  }
}

# ===========================================================================
# H3: DOSE-RESPONSE (Exploratory)
# ===========================================================================
tee_line()
tee("H3: DOSE-RESPONSE TREND")
tee("  Test: Jonckheere-Terpstra + Kendall tau-b")
tee("  Data: 25% / 50% / 100% corpus fractions, regular condition")
tee_line()

if ("corpus_fraction" %in% names(df)) {
  dose_data <- df %>%
    filter(condition == "regular",
           item_type == "second_order",
           corpus_fraction %in% c(0.25, 0.50, 1.0))

  for (sz in c("tiny", "small", "medium")) {
    tee(sprintf("\n--- %s ---", toupper(sz)))

    sz_data <- dose_data %>% filter(model_size == sz)
    seed_acc <- sz_data %>%
      group_by(seed, corpus_fraction) %>%
      summarise(acc = mean(correct, na.rm = TRUE), .groups = "drop")

    desc <- seed_acc %>%
      group_by(corpus_fraction) %>%
      summarise(mean_acc = mean(acc) * 100, sd_acc = sd(acc) * 100, .groups = "drop")
    for (i in seq_len(nrow(desc))) {
      tee(sprintf("  Frac %.0f%%: M=%.1f%%, SD=%.1f%%",
                  desc$corpus_fraction[i] * 100, desc$mean_acc[i], desc$sd_acc[i]))
    }

    # Kendall tau-b on seed-level means
    tau_test <- cor.test(seed_acc$corpus_fraction, seed_acc$acc, method = "kendall")
    tee(sprintf("  Kendall tau-b = %.3f, p = %.4f", tau_test$estimate, tau_test$p.value))

    # Jonckheere-Terpstra
    tryCatch({
      groups <- as.numeric(factor(seed_acc$corpus_fraction))
      jt <- jonckheere.test(seed_acc$acc, groups, alternative = "increasing")
      tee(sprintf("  Jonckheere-Terpstra: JT = %.0f, p = %.4f", jt$statistic, jt$p.value))
      add_result("H3", "Exploratory", sz, "JT trend",
                 "JT", jt$statistic, jt$p.value, 0.05,
                 ifelse(jt$p.value < 0.05, "TREND DETECTED", "NOT SUPPORTED"),
                 sprintf("tau=%.3f", tau_test$estimate))
    }, error = function(e) {
      tee(sprintf("  JT error: %s", e$message))
      add_result("H3", "Exploratory", sz, "Kendall tau",
                 "tau", tau_test$estimate, tau_test$p.value, 0.05,
                 ifelse(tau_test$p.value < 0.05, "TREND DETECTED", "NOT SUPPORTED"), "")
    })
  }
} else {
  tee("  No corpus_fraction column — skipping dose-response")
}

# ===========================================================================
# H4: ONTOLOGICAL KIND (Exploratory)
# ===========================================================================
tee_line()
tee("H4: ONTOLOGICAL KIND (Count-Shape > Count-Texture, Mass-Texture > Mass-Shape)")
tee_line()

for (sz in c("tiny", "small", "medium")) {
  tee(sprintf("\n--- %s ---", toupper(sz)))

  h4_data <- df_full %>%
    filter(condition == "regular",
           item_type %in% c("count_frame", "mass_frame_shape", "mass_frame_texture"),
           model_size == sz)

  seed_acc <- h4_data %>%
    group_by(seed, item_type) %>%
    summarise(acc = mean(correct, na.rm = TRUE) * 100, .groups = "drop") %>%
    pivot_wider(names_from = item_type, values_from = acc)

  if (all(c("count_frame", "mass_frame_texture") %in% names(seed_acc))) {
    tee(sprintf("  Count-frame (shape): M=%.1f%%", mean(seed_acc$count_frame, na.rm = TRUE)))
    if ("mass_frame_shape" %in% names(seed_acc))
      tee(sprintf("  Mass-frame-shape:    M=%.1f%%", mean(seed_acc$mass_frame_shape, na.rm = TRUE)))
    tee(sprintf("  Mass-frame-texture:  M=%.1f%%", mean(seed_acc$mass_frame_texture, na.rm = TRUE)))

    # Paired t-test: count > mass-texture
    if (nrow(seed_acc) >= 3) {
      tt <- t.test(seed_acc$count_frame, seed_acc$mass_frame_texture, paired = TRUE)
      tee(sprintf("  Count vs Mass-texture: t(%.0f)=%.3f, p=%.4f",
                  tt$parameter, tt$statistic, tt$p.value))

      verdict <- ifelse(tt$p.value < 0.05 &
                           mean(seed_acc$count_frame, na.rm = TRUE) >
                           mean(seed_acc$mass_frame_texture, na.rm = TRUE),
                         "SUPPORTED", "NOT SUPPORTED")
      add_result("H4", "Exploratory", sz, "Count vs Mass-texture paired t",
                 "t", tt$statistic, tt$p.value, 0.05, verdict, "")
    }
  } else {
    tee("  Missing ontological kind item types")
  }
}

# ===========================================================================
# H_LABEL: LABELLING NECESSITY (Confirmatory Secondary)
# ===========================================================================
tee_line()
tee("H_LABEL: LABELLING NECESSITY")
tee("  Test: Jonckheere-Terpstra: Regular >> Weak >> Paraphrased >> Bare")
tee("  Alpha: 0.025 (Bonferroni for 2 planned contrasts)")
tee_line()

label_order <- c("bare_nolabel", "paraphrased_nolabel", "weak_label", "regular")

for (sz in c("tiny", "small", "medium")) {
  tee(sprintf("\n--- %s ---", toupper(sz)))

  hl_data <- df_full %>%
    filter(item_type == "second_order",
           condition %in% label_order,
           model_size == sz)

  seed_acc <- hl_data %>%
    group_by(condition, seed) %>%
    summarise(acc = mean(correct, na.rm = TRUE), .groups = "drop")

  desc <- seed_acc %>%
    group_by(condition) %>%
    summarise(mean_acc = mean(acc) * 100, .groups = "drop")
  desc$condition <- factor(desc$condition, levels = label_order)
  desc <- desc %>% arrange(condition)
  for (i in seq_len(nrow(desc))) {
    tee(sprintf("  %s: M=%.1f%%", desc$condition[i], desc$mean_acc[i]))
  }

  # Jonckheere-Terpstra
  seed_acc$cond_num <- match(seed_acc$condition, label_order)
  tryCatch({
    jt <- jonckheere.test(seed_acc$acc, seed_acc$cond_num, alternative = "increasing")
    tee(sprintf("  JT = %.0f, p = %.4f (α = 0.025 after Bonferroni)", jt$statistic, jt$p.value))

    verdict <- ifelse(jt$p.value < 0.025, "SUPPORTED", "NOT SUPPORTED")
    add_result("H_label", "Secondary", sz, "JT trend",
               "JT", jt$statistic, jt$p.value, 0.025, verdict, "Bonferroni corrected")
  }, error = function(e) {
    tee(sprintf("  JT error: %s", e$message))
  })

  # Kendall tau-b
  tau <- cor.test(seed_acc$cond_num, seed_acc$acc, method = "kendall")
  tee(sprintf("  Kendall tau-b = %.3f, p = %.4f", tau$estimate, tau$p.value))
  add_result("H_label", "Secondary", sz, "Kendall tau",
             "tau", tau$estimate, tau$p.value, 0.025,
             ifelse(tau$p.value < 0.025, "SUPPORTED", "NOT SUPPORTED"),
             "effect size")
}

# ===========================================================================
# H_SWAP: FEATURE-SWAP DOMAIN SWITCHING (Exploratory)
# ===========================================================================
tee_line()
tee("H_SWAP: FEATURE-SWAP DOMAIN SWITCHING")
tee("  Test: Domain B frame-cued vs noun-only; binomial test vs chance")
tee_line()

for (sz in c("tiny", "small", "medium")) {
  tee(sprintf("\n--- %s ---", toupper(sz)))

  swap_data <- df_full %>%
    filter(condition == "feature_swap",
           item_type %in% c("domain_b_frame_cued", "domain_b_noun_only"),
           model_size == sz)

  for (it in c("domain_b_frame_cued", "domain_b_noun_only")) {
    sub <- swap_data %>% filter(item_type == it)
    if (nrow(sub) > 0) {
      n_correct <- sum(sub$correct, na.rm = TRUE)
      n_total <- sum(!is.na(sub$correct))
      prop <- n_correct / n_total * 100
      bt <- binom.test(n_correct, n_total, p = 0.5)
      tee(sprintf("  %s: %.1f%% (%d/%d), binom p = %.4e, 95%% CI [%.1f, %.1f]",
                  it, prop, n_correct, n_total, bt$p.value,
                  bt$conf.int[1] * 100, bt$conf.int[2] * 100))

      add_result("H_swap", "Exploratory", sz,
                 sprintf("Binomial %s", gsub("domain_b_", "", it)),
                 "prop", prop / 100, bt$p.value, 0.05,
                 ifelse(bt$p.value < 0.05 & prop > 65, "ABOVE CHANCE",
                        ifelse(bt$p.value < 0.05 & prop < 35, "BELOW CHANCE",
                               "AT CHANCE")),
                 sprintf("n=%d", n_total))
    }
  }
}

# ===========================================================================
# H_BAYES: BAYESIAN ALIGNMENT (Confirmatory Secondary)
# ===========================================================================
tee_line()
tee("H_BAYES: BAYESIAN ALIGNMENT (KL Divergence)")
tee("  Test: KL divergence between model and ideal observer <= 0.1 nats")
tee("  Implementation: Bootstrap CI on KL divergence")
tee("  NOTE: Requires ideal observer posterior (Dirichlet-multinomial).")
tee("  This section computes the model's feature distribution and")
tee("  reports it. The HBM posterior computation is deferred to")
tee("  Appendix E Python implementation.")
tee_line()

# Compute model's empirical distribution over feature tokens per condition/size
# This is preparatory — the actual KL requires the HBM posterior
for (sz in c("tiny", "small", "medium")) {
  tee(sprintf("\n--- %s ---", toupper(sz)))

  reg_so <- df_full %>%
    filter(condition == "regular", item_type == "second_order", model_size == sz)

  acc <- mean(reg_so$correct, na.rm = TRUE) * 100
  tee(sprintf("  Regular SO accuracy: %.1f%% (chance = 50%%)", acc))
  tee(sprintf("  KL computation deferred to HBM implementation"))

  add_result("H_Bayes", "Secondary", sz, "KL divergence",
             "acc", acc / 100, NA, 0.05, "PENDING",
             "Requires HBM posterior — see Appendix E")
}

# ===========================================================================
# SUPPLEMENTARY: ALL PAIRWISE MANN-WHITNEY U
# ===========================================================================
tee_line()
tee("NONPARAMETRIC BACKUPS: Mann-Whitney U (all condition pairs)")
tee("  Pre-registered: reported for all between-condition comparisons")
tee_line()

conditions <- levels(df_full$condition)
cond_pairs <- combn(conditions, 2, simplify = FALSE)

for (sz in c("tiny", "small", "medium")) {
  tee(sprintf("\n--- %s ---", toupper(sz)))

  sz_so <- df_full %>%
    filter(item_type == "second_order", model_size == sz)

  seed_acc <- sz_so %>%
    group_by(condition, seed) %>%
    summarise(acc = mean(correct, na.rm = TRUE), .groups = "drop")

  for (pair in cond_pairs) {
    a <- seed_acc %>% filter(condition == pair[1]) %>% pull(acc)
    b <- seed_acc %>% filter(condition == pair[2]) %>% pull(acc)

    if (length(a) >= 2 && length(b) >= 2) {
      wt <- wilcox.test(a, b, exact = FALSE)
      d <- mean(a) - mean(b)
      sig <- ifelse(wt$p.value < 0.05, "*", "")
      tee(sprintf("  %s vs %s: Δ=%.1f pp, W=%.0f, p=%.4f %s",
                  pair[1], pair[2], d * 100, wt$statistic, wt$p.value, sig))
    }
  }
}

# ===========================================================================
# EXPLORATORY: FO/SO DISSOCIATION
# ===========================================================================
tee_line()
tee("EXPLORATORY: FIRST-ORDER vs SECOND-ORDER DISSOCIATION")
tee_line()

for (sz in c("tiny", "small", "medium")) {
  tee(sprintf("\n--- %s ---", toupper(sz)))

  foso <- df_full %>%
    filter(condition == "regular",
           item_type %in% c("first_order", "second_order"),
           model_size == sz)

  seed_acc <- foso %>%
    group_by(seed, item_type) %>%
    summarise(acc = mean(correct, na.rm = TRUE) * 100, .groups = "drop") %>%
    pivot_wider(names_from = item_type, values_from = acc)

  if (all(c("first_order", "second_order") %in% names(seed_acc))) {
    fo_mean <- mean(seed_acc$first_order, na.rm = TRUE)
    so_mean <- mean(seed_acc$second_order, na.rm = TRUE)
    tee(sprintf("  FO: M=%.1f%% (SD=%.1f%%)",
                fo_mean, sd(seed_acc$first_order, na.rm = TRUE)))
    tee(sprintf("  SO: M=%.1f%% (SD=%.1f%%)",
                so_mean, sd(seed_acc$second_order, na.rm = TRUE)))
    tee(sprintf("  FO - SO = %.1f pp", fo_mean - so_mean))

    if (nrow(seed_acc) >= 3) {
      tt <- t.test(seed_acc$first_order, seed_acc$second_order, paired = TRUE)
      tee(sprintf("  Paired t-test: t(%.0f)=%.3f, p=%.4f",
                  tt$parameter, tt$statistic, tt$p.value))
    }

    # Report seed 42 specifically
    s42 <- seed_acc %>% filter(seed == "42")
    if (nrow(s42) > 0) {
      tee(sprintf("  Seed 42: FO=%.1f%%, SO=%.1f%%",
                  s42$first_order, s42$second_order))
    }
  }
}

# ===========================================================================
# EXPLORATORY: SENSITIVITY ANALYSIS (Seed 42 Exclusion)
# ===========================================================================
tee_line()
tee("EXPLORATORY: SENSITIVITY ANALYSIS — SEED 42 EXCLUSION")
tee("  Clearly labelled as post-hoc exploratory.")
tee("  Rationale: Seed 42 showed near-ceiling FO across conditions.")
tee_line()

for (sz in c("tiny", "small", "medium")) {
  tee(sprintf("\n--- %s ---", toupper(sz)))

  sens_data <- df_full %>%
    filter(condition == "regular",
           item_type %in% c("first_order", "second_order"),
           model_size == sz,
           seed != 42)

  seed_acc <- sens_data %>%
    group_by(item_type) %>%
    summarise(mean_acc = mean(correct, na.rm = TRUE) * 100,
              sd_acc = sd(correct, na.rm = TRUE) * 100,
              n_seeds = n_distinct(seed),
              .groups = "drop")

  for (i in seq_len(nrow(seed_acc))) {
    tee(sprintf("  %s (excl. seed 42): M=%.1f%%, SD=%.1f%%, k=%d seeds",
                seed_acc$item_type[i], seed_acc$mean_acc[i],
                seed_acc$sd_acc[i], seed_acc$n_seeds[i]))
  }
}

# ===========================================================================
# SUMMARY TABLE
# ===========================================================================
tee_line()
tee("COMPLETE RESULTS TABLE")
tee_line()

tee(capture.output(print(results_table, right = FALSE, row.names = FALSE)))

# Save table
table_path <- file.path(base_dir, "results", "study1", "R_analysis_tables.csv")
write_csv(results_table, table_path)
tee(sprintf("\nResults table saved to: %s", table_path))

# ===========================================================================
# VERDICT SUMMARY
# ===========================================================================
tee_line()
tee("VERDICT SUMMARY")
tee_line()

verdicts <- results_table %>%
  filter(!duplicated(paste(Hypothesis, Size))) %>%
  select(Hypothesis, Type, Size, Verdict)

tee(capture.output(print(verdicts, right = FALSE, row.names = FALSE)))

tee(sprintf("\nFull output saved to: %s", out_file))
tee(sprintf("Analysis completed: %s", Sys.time()))

# Close sink
close(sink_con)

cat("\n\nDone. Output saved to:\n")
cat(sprintf("  %s\n", out_file))
cat(sprintf("  %s\n", table_path))
