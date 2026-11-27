# weights/

Aquí van los pesos de los modelos (fuera de git). Se descargan y verifican con:

```bash
# u100, rs_ca y rs_ca_tr
python scripts/download_weights.py
# También e6 (opcional)
python scripts/download_weights.py --all
```

`registry.json` (en git) lista cada fichero con su tamaño, SHA256, licencia y URL. Los
pesos son `state_dict` en float32 (`torch.save`) y se cargan con
`janonet.models.load_model("<modelo>", "weights/<modelo>.pt")`.

Licencia de los pesos: CC BY-NC 4.0, sujeta a los términos de uso de los datasets de
entrenamiento (ver `MODEL_CARD.md` y `DATA.md`).
