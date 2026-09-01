"""
Tiny ComfyUI custom-node package whose only job is to ship a web extension
that opens the GDS->SEM workflow on startup.

It registers no nodes — WEB_DIRECTORY is the whole point: ComfyUI serves
everything under ./js at /extensions/gds2sem_default_workflow/, and loads
any .js it finds there into the frontend.
"""

WEB_DIRECTORY = "./js"
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
