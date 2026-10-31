# Soup Kernel Challenge — решение команды `<TEAM>`

`soup train` (Soup 0.77.0, Qwen2.5-1.5B-Instruct, NF4 + LoRA r=16 на q/v, batch 4, до 512 токенов,
64 шага) на бесплатной Kaggle T4 — быстрее и с кратно меньшей пиковой памятью, на тех же данных,
с теми же настройками, тем же адаптером и теми же выходами Soup. Восемь своих Triton-кернелов
(`@triton.jit`, папка `solution/`) выполняют работу в каждом шаге обучения.

- **Kaggle-ноутбук (сохранённая версия):** `<ВСТАВИТЬ ССЫЛКУ>`
- **Финальный коммит:** `<ВСТАВИТЬ ССЫЛКУ НА КОММИТ>`
- Видео / презентация: `<ссылки>`

## Результаты (Kaggle T4, наш запуск `harness/run_task.py`)

| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Время всей команды `soup train`, с | 88.03 | **56.29** | speedup **x1.564** |
| 64 шага обучения, с | 55.35 | 33.26 | x1.664 |
| Пик памяти GPU (nvidia-smi), MiB | 14 345 | **3 299** | base/ours **x4.348** |
| Eval loss (отложенные строки 40000–40255) | 2.04963 | 2.04970 | +0.003 % |

Кривая loss: первый шаг отличается от базы на 0,01%, среднее по всем шагам — на 0,01%.

Доля наших кернелов во времени шагов 3+ (check 4, CUDA-события харнесса):

| Кернел | Доля |
|---|---|
| `nf4_unpack_kernel` | 8.9% |
| `gated_silu_bwd_inplace_kernel` | 1.7% |
| `gated_silu_fwd_kernel` | 1.0% |
| `rmsnorm_bwd_kernel` | 1.0% |
| `rmsnorm_fwd_kernel` | 0.9% |
| `head_stats_grad_kernel` | 0.6% |
| `rope_pack_fwd_kernel` | 0.3% |
| `rope_pack_bwd_kernel` | 0.2% |
| **всего наших** | **14.8%** |

Проверки `harness/check_rules.py` — все 10 PASS:

| Проверка | Результат |
|---|---|
| 1. все запуски базы и решения завершились | PASS |
| 2. нет готовых кернел-библиотек | PASS: none loaded / imported, packages unchanged |
| 3. свои `@triton.jit` из `solution/` работают в обучении | PASS: все 8 кернелов |
| 4. свои кернелы ≥ 5% времени обучения | PASS: 14.8% |
| 5. та же задача и выходы Soup (шаги, адаптер, настройки, токены, метрики, rewind) | PASS: identical within tolerance |
| 6. кривая loss следует базе | PASS: first step 0.01%, mean 0.01% |
| 7. качество: eval loss не выше базы более чем на 0.5% | PASS: +0.003% |
| 8. не хуже базы по времени и памяти | PASS: time x0.639, memory x0.230 |
| 9. чистый замер (T4, свободные GPU, одна GPU, аллокатор харнесса) | PASS |
| 10. регрессионные тесты Soup с нашими патчами | PASS: 79 passed |

Версии: Python 3.13.15, torch 2.11.0+cu128, triton 3.6.0, transformers 5.16.1, peft 0.20.0,
trl 0.29.1, bitsandbytes 0.50.2, accelerate 1.14.0, soup-cli 0.77.0, 2× Tesla T4 (используется одна).
Один прогон каждой стороны (`run_task.py` по умолчанию); жюри запускает `--repeats 2`
(база, решение, решение, база), разброс на T4 по ТЗ — 1–4%. Таблицы печатает
`python tools/results_table.py`.

Куда ушли ~11 ГБ: полные logits словаря и их fp32-копии/градиенты (~4 ГБ), fp32-активации MLP
(~4–6 ГБ), паддинг-токены во всех слоях, fp32-таблица эмбеддинга (0,47 ГБ постоянно +
0,47 ГБ временной копии). Остаток ~3,3 ГБ — контекст CUDA, 4-битные веса, fp16-эмбеддинг,
активации attention и LoRA.

## Что сделано и почему

Профиль базового шага Soup на T4 (из ТЗ): ~44% GPU — мелкие поэлементные операции и приведения
типов, ~36% — матричные умножения, ~7% — softmax/CE по словарю 151 936, ~5% — attention, ~5% —
распаковка NF4. Пик ~14,3 ГБ из 15. Разбор исходников Soup/transformers/peft/bitsandbytes показал,
откуда это берётся:

1. **Остаточный поток и все проекции живут в fp32.** `prepare_model_for_kbit_training` переводит
   нормы в fp32, нормы выдают fp32, а `bnb.nn.Linear4bit` возвращает результат в dtype входа —
   значит gate/up/q/k/v приходят из fp16-GEMM и тут же копируются в fp32. Только MLP держит для
   backward три fp32-тензора `[токены, 8960]` на слой (≈220 МБ/слой при 2048 токенах, ≈6 ГБ на 28
   слоёв), и все SiLU/умножения/касты идут по ним в fp32.
2. **Logits словаря.** fp16-logits `[4, 512, 151936]` (622 МБ) → accelerate копирует в fp32
   (1,2 ГБ) → loss апкастит и хранит log-softmax (1,2 ГБ) → TRL делает сдвинутую копию для
   accuracy и считает энтропию → rewind Soup считает ещё одну CE по строкам → backward строит
   градиент той же формы в fp32 и fp16. ≈4 ГБ на пике и ~7 проходов по logits.
3. **Паддинг.** Батч дополняется до самой длинной строки (медиана 140 токенов, максимум 512), то
   есть ~30–40% токенов — паддинг, который считается во всех GEMM, нормах и MLP и выбрасывается.
4. **Эмбеддинг/lm_head (связанные веса) в fp32** — 0,93 ГБ, а под autocast каждую итерацию
   делается ещё fp16-копия 0,47 ГБ для головы.

### Что мы заменили (всё — `solution/`, подключается одной `apply()`)

| Кернел (`@triton.jit`) | Файл | Что заменяет |
|---|---|---|
| `head_stats_grad_kernel` | `head.py` | весь путь logits → loss/энтропия/accuracy/rewind/градиент |
| `gated_silu_fwd_kernel`, `gated_silu_bwd_inplace_kernel` | `mlp.py` | fp32 SiLU·up и его backward |
| `rmsnorm_fwd_kernel`, `rmsnorm_bwd_kernel` | `rmsnorm.py` | residual add + Qwen2RMSNorm (6+ ops в fp32) |
| `rope_pack_fwd_kernel`, `rope_pack_bwd_kernel` | `rope.py` | LoRA-add + RoPE + transpose + касты q/k/v |
| `nf4_unpack_kernel` | `nf4.py` | распаковка NF4 (bitsandbytes, 2 запуска) |

**1. Голова без полных logits (`head.py`).** Только реальные токены, кусками по 256 строк
(сначала строки с меткой). Для куска: cuBLAS `logits = h·Wᵀ` (fp16, ровно тот GEMM, что делает
база под autocast) → Triton, одна программа на строку, два прохода по 151 936 значениям:
проход 1 — онлайн-softmax в fp32 (бегущий максимум, Σeᶻ, Σeᶻ·z и первый argmax) даёт lse,
энтропию `lse − Σp·z`, loss токена `lse − z[label]` и попадание argmax; проход 2 (только строки с
меткой) — на месте пишет градиент `(softmax − onehot)·scale` в fp16 → cuBLAS `dh = dlogits·W`.
Logits куска сразу освобождаются: вместо ≈4 ГБ — один буфер `[256, 151936]` fp16 (78 МБ).
Градиент считается в forward (голова заморожена) и масштабируется на loss scale GradScaler в
backward; масштаб fp16-градиента выбран степенью двойки ≈ как у базы (65536/токены), поэтому
перемасштабирование точное. Значения для TRL (энтропия, `mean_token_accuracy`, `num_tokens`) и
для `rewind.jsonl` (средняя CE по строке, число токенов) считаются по тем же формулам, что в
TRL 0.29.1 и Soup, из этих же чисел (`trainer_patch.py`).

**2. MLP (`mlp.py`).** gate и up — один GEMM против обоих весов, распакованных в один буфер
`[17920, 1536]`; Triton-кернел читает fp16 `[токены, 2·8960]` и пишет `silu(g)·u` в fp16
(вычисление в fp32; fp32-копии базы не содержали лишней информации — это были апкасты fp16-
выходов GEMM, а произведение база округляет в fp16 в том же месте, перед down_proj). Для backward
сохраняется только вход `[токены, 1536]` fp16: gate/up пересчитываются одним GEMM, второй
кернел на месте превращает их в градиенты gate/up, и градиент входа — снова один GEMM.
Флаг `RECOMPUTE_GATE_UP` (по умолчанию `True`) выбирает пересчёт (меньше памяти) или хранение
gate/up в fp16.

**3. Add + RMSNorm (`rmsnorm.py`).** Одна программа на токен: `s = x + delta` (fp32 остаточный
поток), `y = w·s·rsqrt(mean(s²)+eps)`, запись `y` в fp16 для GEMM и — для входной нормы слоя —
ещё в fp32 в паддинговую раскладку `[batch·seq, 1536]`, которую читает LoRA-dropout. Backward —
одна программа на токен: складывает в fp32 градиенты GEMM-пути (fp16) и LoRA-пути (fp32),
добавляет остаточный градиент.

**4. Q/K/V (`rope.py`).** q, k, v — один GEMM `[Wq;Wk;Wv]` (+bias). Triton-программа на токен
читает строку `[2048]`, добавляет LoRA-выходы q и v в fp32, вращает q и k (RoPE) в fp32 и пишет
fp16 прямо в нулевые тензоры `[batch, heads, seq, 128]` — раскладку, которую ест SDPA (у базы
это ~7 операций на тензор в fp32 плюс касты autocast). Backward — транспонированный поворот;
LoRA-ветки получают view того же буфера (у базы обе ветки получают тот же fp16-каст fp32-
градиента).

**5. NF4 (`nf4.py`).** Одна Triton-программа распаковывает 1024 байта в 2048 fp16, вычисляя
вложенный absmax (`code2[u8]·absmax2 + offset`) на лету; несколько проекций пишутся в один буфер
(q/k/v, gate/up), чтобы GEMM был один. Арифметика та же, что у bitsandbytes (без FMA-слияния:
`enable_fp_fusion=False`), — значения побитно совпадают (тест `test_nf4_unpack_matches_bitsandbytes`).

**6. Без pad-токенов (`model.py`).** Нормы, все проекции, MLP и голова идут только по реальным
токенам (`attention_mask == 1`); attention — по паддинговой раскладке, как у базы (SDPA из
transformers, та же маска). У pad-токенов в базе метки −100, их ключи замаскированы, энтропия TRL
умножается на маску, поэтому их выбрасывание не меняет ни loss, ни градиенты, ни метрики.
Строки разных батчей не смешиваются.

**7. Эмбеддинг в fp16.** Чекпойнт грузится на T4 в fp16, `prepare_model_for_kbit_training`
делает из него fp32 — значит fp16-копия точна (проверяется поэлементно перед заменой). Выход
эмбеддинга приводится обратно к fp32 (хук), так что все остальные пути видят то же, что раньше.
Голова использует ту же fp16-таблицу, что под autocast кастовала база. −0,47 ГБ постоянно и
−0,47 ГБ временной копии на каждом шаге.

**8. Фоновая компиляция (`warmup.py`).** Кэш Triton в каждом замере пустой, а время компиляции
входит в замер. Константы кернелов известны из `config.json` модели (она уже скачана харнессом),
поэтому компиляция всех кернелов запускается в фоновом потоке *того же процесса* перед загрузкой
модели и идёт параллельно с ней (никаких сторонних процессов и готовых кэшей). Если поток не
успел, кернелы компилируются при первом вызове, как обычно. Читается только `config.json` той же
модели, которую затем грузит Soup (из локального кэша HF, без сети); это поток внутри процесса
`soup train`, а не фоновый процесс, и его время входит в замер.

### Почему качество и выходы Soup не меняются

- Те же модель, данные, порядок батчей, шаги, LoRA (r, alpha, dropout 0.05, q/v), оптимизатор,
  расписание, переключатели Soup — мы ничего из этого не трогаем; патчи на классах
  (`Qwen2ForCausalLM.forward`, `trl.SFTTrainer.compute_loss`, `rewind_hf.row_losses`,
  `SFTTrainerWrapper._setup_transformers`), каждый уходит в оригинал для всего, что не наш случай.
- **LoRA-dropout получает тот же тензор** (та же форма `[B,T,1536]`, fp32, те же значения), что и
  в базе, в том же порядке (q, затем v в каждом слое) — значит, генератор выдаёт те же маски.
- Округления в fp16 происходят там же, где их делает база (autocast/bitsandbytes); всё
  промежуточное — в fp32.
- Тест `solution/test_model.py` сравнивает наш шаг с непатченной моделью (4-bit Qwen2 + LoRA, те
  же сид и dropout): loss совпадает до 1e-6 отн., энтропия совпадает, per-row loss rewind — до
  1e-5, градиенты LoRA — до ~1e-3 отн. (шум fp16). `solution/test_trainer.py` гоняет настоящий
  `SFTTrainer` TRL с rewind-подклассом Soup: те же loss, `entropy`, `num_tokens`,
  `mean_token_accuracy`, строки и токены rewind.

### Источники идей (код написан нами)

- Градиент головы в forward кусками, без полных logits: Liger Kernel `FusedLinearCrossEntropy`
  (https://github.com/linkedin/Liger-Kernel) и Cut Cross-Entropy (Wijmans et al., 2024,
  arXiv:2411.09009). В Soup есть похожий режим на PyTorch (`stream_loss_chunk`) — мы его **не
  включаем**; наша голова — отдельная реализация на Triton, считающая ещё энтропию, argmax и
  per-row loss за тот же проход.
- Онлайн-softmax: Milakov & Gimelshein, 2018, arXiv:1805.02867.
- Построчная норма forward/backward: туториал Triton «Layer Normalization»
  (https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html).
- Подключение через патчи классов модели: Unsloth (https://github.com/unslothai/unsloth).
- Формат NF4 и двойная квантизация: QLoRA (Dettmers et al., 2023, arXiv:2305.14314), раскладка
  байтов — по исходникам bitsandbytes 0.50.2.
- RoPE: Su et al., RoFormer, arXiv:2104.09864; формулы — `apply_rotary_pos_emb` из transformers.

## Как запустить

### Kaggle (так проверяет жюри)

1. Kaggle → **Create → New Notebook → File → Import Notebook** → `templates/kaggle_notebook.ipynb`
   из этого репозитория. Справа: **Accelerator: GPU T4 x2**, **Internet: on**.
2. В первой ячейке: `REPO_URL = "<URL этого репозитория>.git"`, `COMMIT = "<финальный коммит>"`.
   Для приватного репозитория — секрет `GITHUB_TOKEN` (Add-ons → Secrets).
3. **Run All** (~10–15 минут). Ноутбук ставит `soup-cli[train]==0.77.0` и закреплённые версии,
   готовит данные, гоняет наши тесты на GPU (`python -m pytest solution -q`), затем
   `harness/run_task.py` (база и решение), `check_rules.py` и `score.py`.
4. Таблицы для README: `!cd /kaggle/working/repo && python tools/results_table.py`.
5. Убедиться, что быстрый путь реально работал: в `runs/solution_1/metrics.json` поле
   `triton_launches` → `head_stats_grad_kernel` должно быть порядка сотен (≈ число шагов × кусков),
   а не 1 (одиночный запуск — это только фоновая компиляция); check 4 показывает долю кернелов.

Для итоговой проверки как у жюри: `python harness/run_task.py --repeats 2` (база, решение,
решение, база).

### Локально (Linux + CUDA)

```bash
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" \
    "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
python task/prepare_data.py
python -m pytest solution -q            # наши кернелы и весь шаг против PyTorch / непатченной модели
python harness/run_task.py              # -> runs/result.json
python harness/check_rules.py           # -> runs/checks.json
python tools/results_table.py           # таблицы для README
```

Без GPU тесты идут в CPU-интерпретаторе Triton: `TRITON_INTERPRET=1 python -m pytest solution -q`
(11 тестов: каждый кернел forward/backward против PyTorch, весь шаг против непатченной модели,
связка с TRL/rewind). Регрессионные тесты Soup с нашими патчами:
`SOUP_APPLY_SOLUTION=1 python -m pytest tests -q` → 79 passed.

## Ограничения и что дальше

- **Attention не наш.** SDPA (mem-efficient) остаётся на паддинговой раскладке и хранит q и
  размноженные по GQA k/v (×6) в fp16 — теперь это крупнейшая часть активаций (~27 МБ/слой при
  2048 позициях). Следующий шаг — свой Triton flash-attention с GQA и varlen без паддинга.
- **GEMM — cuBLAS по распакованным весам.** Распаковка NF4 своя и быстрая, но вес всё равно
  материализуется в fp16 на время GEMM (до 55 МБ). Слитый NF4-GEMM на Triton для sm75 мог бы
  убрать это и часть трафика.
- **Пересчёт gate/up** экономит ~67 МБ/слой ценой одного лишнего GEMM на слой в backward;
  `RECOMPUTE_GATE_UP = False` даёт обратный компромисс.
- LoRA-ветки (rank 16) и dropout остаются в PyTorch, чтобы маски dropout совпадали с базой.
- Быстрый путь включается только для этой конфигурации (Qwen2, bitsandbytes NF4 с двойной
  квантизацией, ванильная LoRA на q/v, SDPA, fp16 autocast); всё остальное идёт в исходный код
  без изменений. На Ampere+ (bf16) путь не проверялся и не включается (нужен fp16 autocast).
- Числа в fp16 не побитно равны базе (другие формы GEMM, суммирование в fp32 вместо fp16 в паре
  мест) — расхождения на уровне шума fp16, проверки 5–7 это допускают.

## Структура

| Путь | Что это |
|---|---|
| `solution/__init__.py` | `apply()`: ставит четыре патча |
| `solution/trainer_patch.py` | патчи forward / TRL compute_loss / rewind / setup (fp16-эмбеддинг, warmup) |
| `solution/model.py` | обучающий forward Qwen2 + LoRA на реальных токенах |
| `solution/head.py`, `mlp.py`, `rmsnorm.py`, `rope.py`, `nf4.py` | кернелы и их autograd |
| `solution/warmup.py` | фоновая компиляция кернелов во время загрузки модели |
| `solution/test_*.py` | тесты: кернелы, весь шаг, связка с TRL/rewind |
| `tools/results_table.py` | таблицы результатов из `runs/` |
| `harness/`, `tests/`, `task/`, `templates/` | стартовый набор организаторов, без изменений |

Лицензия Apache-2.0 (`LICENSE`, `NOTICE`); данные и модель — см. README стартового набора
(AmanMussa/kazakh-instruction-v2, Qwen/Qwen2.5-1.5B-Instruct).
