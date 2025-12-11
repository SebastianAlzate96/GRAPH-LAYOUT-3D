# Layout de Grafos en 3D con Modelos de Fuerzas

Proyecto final del curso **COMP6838 — Visualización de Grafos**.

Este repositorio implementa un sistema interactivo en 3D para el *layout* de grafos usando tres modelos de fuerzas:

- **Fruchterman–Reingold (FR)**
- **LinLog**
- **Spring–Electrical (SE)**

El objetivo es obtener layouts estables y legibles en 3D, con control explícito sobre parámetros físicos, temperatura y coloración por comunidades, manteniendo tasas de cuadro interactivas.

---

## Características principales

- Visualización 3D en **OpenGL + GLUT**.
- Tres modelos de fuerzas conmutables en tiempo real:
  - FR (baseline compacto).
  - LinLog (separación de comunidades y puentes).
  - Spring–Electrical (control fino del espaciado).
- Control de:
  - Temperatura y *cooling*.
  - “Reheat” para reacomodar el layout tras cambios bruscos.
  - Suavizado temporal (EMA) para reducir *jitter*.
- Coloración por:
  - Grado del nodo.
  - Comunidades detectadas con **k-means++** sobre posiciones 3D.
- Soporte de grafos:
  - Grafo sintético (anillo + aristas de largo alcance).
  - Grafo clásico **Karate Club de Zachary** cargado desde CSV robusto.
- HUD en pantalla con:
  - FPS, temperatura, Δ medio, modo de fuerza, parámetros de SE.
  - Panel de ayuda con todos los controles.

---