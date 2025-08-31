#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import logging
from typing import Dict, Any, List

class WebInterface:
    def __init__(self, template_folder: str = 'templates'):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.app = Flask(__name__, 
                         template_folder=template_folder,
                         static_folder=project_root,
                         static_url_path='')

        self.socketio = SocketIO(self.app, async_mode='threading')
        self.logger = logging.getLogger(__name__)
        self.status = {'state': 'idle', 'message': '等待開始'}
        self.log_messages = []
        self.memory_usage = {}
        self.cache_stats = {}
        self.tuning_results = []
        self.available_models = []
        self._register_routes()
        self._register_socketio_events()

    def set_available_models(self, models: List[str]):
        self.available_models = models
        self.logger.info("Broadcasting available_models event with data: %s", {'models': self.available_models})
        self.socketio.emit('available_models', {'models': self.available_models})

    def _register_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/api/status')
        def get_status():
            return jsonify(self.status)
        
    def _register_socketio_events(self):
        """註冊 Socket.IO 事件"""
        @self.socketio.on('connect')
        def handle_connect():
            self.logger.info("Web UI 客戶端已連接")
            emit('status_update', self.status)
            emit('log_update', self.log_messages)
            emit('memory_update', self.memory_usage)
            emit('cache_update', self.cache_stats)
            emit('results_update', self.tuning_results)
            emit('available_models', {'models': self.available_models})

        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.logger.info("Web UI 客戶端已斷開")

        @self.socketio.on('start_tuning')
        def handle_start_tuning(data): pass
        
        @self.socketio.on('stop_tuning')
        def handle_stop_tuning(): pass

        @self.socketio.on('generate_report')
        def handle_generate_report(): pass
    
    def run(self, host: str = '127.0.0.1', port: int = 5000, debug: bool = False):
        self.socketio.run(self.app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
    
    def set_status(self, state: str, message: str):
        self.status = {'state': state, 'message': message}
        self.socketio.emit('status_update', self.status)
        self.add_log_message('info', message)
    
    def add_log_message(self, level: str, message: str):
        log_entry = {'level': level, 'message': message}
        self.log_messages.append(log_entry)
        if len(self.log_messages) > 100: self.log_messages.pop(0)
        self.socketio.emit('log_update', [log_entry])
    
    def update_memory_usage(self, data: Dict[str, Any]):
        self.memory_usage = data
        self.socketio.emit('memory_update', self.memory_usage)
    
    def update_cache_stats(self, data: Dict[str, Any]):
        self.cache_stats = data
        self.socketio.emit('cache_update', self.cache_stats)
    
    def add_tuning_result(self, result: Dict[str, Any]):
        self.tuning_results.append(result)
        self.socketio.emit('new_result', result)

web_ui = WebInterface()

def get_web_ui() -> WebInterface:
    return web_ui

def start_web_ui(host: str = '127.0.0.1', port: int = 5000, debug: bool = False):
    global web_ui
    web_ui.run(host=host, port=port, debug=debug)