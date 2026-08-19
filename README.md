# ComfyUI ConvRot Converter per Windows

Interfaccia drag-and-drop per convertire modelli ComfyUI in **INT8 + ConvRot**. La conversione è eseguita dallo script ufficiale [`quant_int8_convrot.py`](https://github.com/Comfy-Org/comfy-model-tools/blob/main/quant_int8_convrot.py) di Comfy-Org.

## Installazione

1. Installa [Python 3.12 a 64 bit](https://www.python.org/downloads/) selezionando anche **Python Launcher**.
2. Fai doppio clic su `setup.bat`. Verrà creata una cartella `.venv` locale e saranno installati PyTorch CUDA, `comfy-kitchen`, `safetensors` e il supporto drag-and-drop.
3. Dalle volte successive avvia semplicemente `AVVIA.bat`.

Il primo setup scarica PyTorch e può richiedere diversi GB. La conversione richiede una GPU NVIDIA compatibile CUDA e sufficiente VRAM; non modifica mai il modello originale.

Il setup ignora gli eventuali mirror aggiuntivi configurati globalmente in `pip` e usa soltanto PyPI e l'indice ufficiale PyTorch CUDA. Questo evita blocchi causati da mirror aziendali o NVIDIA non raggiungibili.

## Uso

1. Trascina uno o più file `.safetensors`, `.pth`, `.pt`, `.ckpt` o `.bin` nell'area azzurra.
2. Lascia vuota la cartella di output per salvare accanto all'originale, oppure scegline una.
3. Per un'architettura nuova, esegui prima **Solo analisi**: mostra quali layer verranno convertiti senza scrivere nulla.
4. Premi **Avvia conversione**.

Nel log, la sezione **source storage dtypes** mostra la precisione reale letta dall'header del modello (`F16`, `BF16`, `F8_E4M3`, `I8`). Non affidarti soltanto al nome del file o alla riga `compute/passthrough dtype`, che descrive invece la precisione scelta per i layer non quantizzati. Se il sorgente è già prevalentemente FP8 o INT8, viene mostrato un avviso esplicito contro la doppia quantizzazione.

### Preset automatico LTX-2.3

I bundle LTX-2.3 a 48 blocchi vengono riconosciuti automaticamente. Il converter applica la selezione protetta Comfy-Org: quantizza tutti i 34 Linear dei blocchi 2–45 (1.496 layer), mantiene in alta precisione i blocchi 0, 1, 46 e 47 e i connettori audio/video, e include i piccoli `to_gate_logits` ignorando `Min GEMM` soltanto per questo preset. Se la struttura non produce esattamente 1.496 layer, la conversione viene fermata anziché creare un file ambiguo.

I text encoder standalone vengono riconosciuti automaticamente con preset protetti:

- **UMT5/UMT5-XXL**: quantizza le proiezioni attention e feed-forward di tutti i blocchi encoder, mantenendo embedding condiviso, normalizzazioni e finali nella precisione sorgente.
- **Gemma**: quantizza soltanto attention/MLP dei blocchi linguistici interni; embedding, vision tower, teste e primo/ultimo blocco restano BF16/FP16.
- **Qwen/Qwen-VL**: applica la stessa protezione conservativa a embedding, componenti visuali, teste e blocchi linguistici estremi.

I preset text encoder eseguono anche un controllo di uniformità strutturale per blocco e interrompono la conversione se il checkpoint non corrisponde alla famiglia riconosciuta. I checkpoint AIO contenenti anche un diffusion model non vengono scambiati per text encoder standalone.

Da riga di comando il comportamento può essere controllato con `--preset auto`, `--preset ltx2_official`, `--preset umt5_text`, `--preset gemma_text`, `--preset qwen_text` o `--preset generic`. La GUI usa `auto`.

Il trascinamento funziona sull'intera finestra, compresa la tabella e il riquadro del log. Su Windows non avviare `AVVIA.bat` come amministratore: per ragioni di sicurezza Windows blocca il drag-and-drop da Esplora file non elevato verso un programma eseguito come amministratore.

L'interfaccia abilita la modalità DPI per-monitor di Windows e adatta automaticamente dimensioni iniziali e colonne allo scaling del desktop (inclusi display 4K al 150–200%).

Il nome viene trasformato come nello script ufficiale: `model_bf16.safetensors` diventa `model_int8_convrot.safetensors`; in assenza di `bf16`, `fp16` o `fp32`, viene aggiunto `_int8_convrot`.

### Opzioni

- **Min GEMM 256**: impostazione ufficiale consigliata; evita layer troppo piccoli, dove INT8 sarebbe solo overhead.
- **MSE clip**: modalità sperimentale che può ridurre l'errore dei pesi; convalidare sempre il risultato.
- **Riduci FP32 residui**: converte in precisione di calcolo alcuni layer FP32 non quantizzati, riducendo lo spazio.
- **Report qualità**: produce un file `.quality.tsv` con errore relativo, similarità coseno e group size di ogni layer.

## Note importanti

- ConvRot è pensato per i layer compatibili rilevati automaticamente. Alcune architetture o loader con rimappatura delle chiavi possono non essere adatti: per questo è utile il dry run.
- Il programma elabora i modelli uno alla volta per limitare il consumo di memoria.
- Se un file di destinazione esiste già, l'app chiede conferma prima di sovrascriverlo.

Lo script Comfy-Org incluso è distribuito secondo GPL-3.0; la copia della licenza è in `LICENSE-COMFY-MODEL-TOOLS`.
