"""
Configuration wizard for Robot Agent
Simple setup page for initial configuration
"""

import os
from pathlib import Path
from nicegui import ui
import logging

logger = logging.getLogger(__name__)


def needs_configuration() -> bool:
    """
    Check if configuration is needed
    Returns True if .env doesn't exist or contains default values
    """
    env_path = Path('.env')
    
    # If no .env file, need configuration
    if not env_path.exists():
        logger.info("No .env file found - needs configuration")
        return True
    
    # Check if any required vars are missing or default
    required_vars = ['LLM_PROVIDER', 'LLM_MODEL']
    default_values = ['CHANGE_ME', 'your-api-key', '', 'none']
    
    try:
        with open(env_path, 'r') as f:
            content = f.read()
            
            # Check if file is empty or has default values
            if not content.strip():
                return True
            
            # Check for default/placeholder values
            content_lower = content.lower()
            for default in default_values:
                if default in content_lower:
                    logger.info(f"Found default value '{default}' in .env - needs configuration")
                    return True
            
            # Check if required vars are present
            for var in required_vars:
                if var not in content:
                    logger.info(f"Missing required var {var} - needs configuration")
                    return True
    
    except Exception as e:
        logger.error(f"Error reading .env: {e}")
        return True
    
    return False


@ui.page('/setup')
def setup_page():
    """Simple setup page for initial configuration"""
    
    # Page styling
    ui.colors(primary='#2563EB')
    
    with ui.column().classes('w-full max-w-2xl mx-auto p-8 gap-6 absolute-center'):
        # Header
        with ui.card().classes('w-full p-6 bg-gradient-to-r from-blue-500 to-purple-600 text-white'):
            ui.label('🔧 Configuration Wizard').classes('text-3xl font-bold')
            ui.label('Welcome to Robot Agent!').classes('text-xl opacity-90 mt-2')
            ui.label('Let\'s get your LLM provider set up.').classes('text-md opacity-80')
        
        # Main config card
        with ui.card().classes('w-full p-6'):
            ui.label('LLM Provider Settings').classes('text-xl font-bold mb-4')
            ui.label('Select which LLM backend you want to use').classes('text-gray-600 mb-6')
            
            # Provider selection
            provider = ui.select(
                {
                    'ollama': '🦙 Ollama (Local, Free)',
                    'lmstudio': '🎯 LM Studio (Local, Free)',
                    'openai': '🤖 OpenAI (Cloud, Paid)'
                },
                value='ollama',
                label='LLM Provider'
            ).classes('w-full mb-4')
            
            # Model name input
            model = ui.input(
                value='llama2',
                label='Model Name'
            ).classes('w-full mb-4')
            model.tooltip('Examples: llama2, mistral, phi (for Ollama) | gpt-3.5-turbo (for OpenAI)')
            
            # Provider-specific help
            with ui.column().classes('w-full bg-blue-50 p-4 rounded mt-2'):
                ui.label('💡 Provider Information:').classes('font-bold')
                help_text = ui.label().classes('text-sm')
                
                def update_help():
                    if provider.value == 'ollama':
                        help_text.set_text(
                            'Ollama: Make sure Ollama is running (ollama serve)\n'
                            'Default model: llama2 (run "ollama pull llama2" first)'
                        )
                    elif provider.value == 'lmstudio':
                        help_text.set_text(
                            'LM Studio: Start the local inference server\n'
                            'Go to "Local Server" tab and click "Start Server"'
                        )
                    else:  # openai
                        help_text.set_text(
                            'OpenAI: You\'ll need an API key\n'
                            'Add OPENAI_API_KEY=sk-... to .env file'
                        )
                
                provider.on('change', update_help)
                update_help()
        
        # Memory settings card
        with ui.card().classes('w-full p-6'):
            ui.label('Memory Settings').classes('text-xl font-bold mb-4')
            
            memory = ui.select(
                {
                    'simple': '📝 Simple (In-memory, no setup)',
                    'faiss': '🔍 FAISS (Vector search, requires pip install faiss-cpu)',
                    'cognee': '🧠 Cognee (Knowledge graph, requires pip install cognee)'
                },
                value='simple',
                label='Memory Backend'
            ).classes('w-full')
        
        # Hardware settings card
        with ui.card().classes('w-full p-6'):
            ui.label('Hardware Settings').classes('text-xl font-bold mb-4')
            
            hardware = ui.switch('Enable hardware device support').classes('mb-2')
            ui.label('(Requires Arduino/ESP32 connected via USB)').classes('text-xs text-gray-500')
        
        # Save button
        def save_config():
            """Save configuration to .env file"""
            try:
                with open('.env', 'w') as f:
                    f.write(f"""# Robot Agent Configuration
# Generated by Configuration Wizard - {provider.value}

# LLM Settings
LLM_PROVIDER={provider.value}
LLM_MODEL={model.value}

# Memory Settings
MEMORY_BACKEND={memory.value}

# Hardware Settings
ENABLE_HARDWARE={'true' if hardware.value else 'false'}

# Audio Settings (defaults)
TTS_ENABLED=true
STT_ENABLED=true
TTS_PROVIDER=pyttsx3
STT_PROVIDER=whisper
""")
                
                ui.notify('✅ Configuration saved successfully!', type='positive', position='top')
                ui.notify('Restarting application...', type='info', position='top')
                
                # Reload after 2 seconds
                ui.timer(2.0, lambda: ui.open('/'), once=True)
                
            except Exception as e:
                ui.notify(f'❌ Error saving configuration: {e}', type='negative')
        
        # Save button
        with ui.row().classes('w-full gap-4 mt-4'):
            ui.button('💾 Save Configuration', on_click=save_config).props('color=primary size=lg').classes('flex-1')
            ui.button('❌ Cancel', on_click=lambda: ui.open('/')).props('flat').classes('flex-1')
        
        # Footer
        ui.label('You can change these settings later by editing the .env file').classes('text-xs text-gray-500 mt-4')


# Auto-create default .env if it doesn't exist
def ensure_env_file():
    """Create default .env if it doesn't exist"""
    env_path = Path('.env')
    if not env_path.exists():
        with open(env_path, 'w') as f:
            f.write("""# Robot Agent Configuration
# Uncomment and modify the lines below

# LLM Settings (choose one)
LLM_PROVIDER=ollama
LLM_MODEL=llama2

# For OpenAI (uncomment and add your key)
# LLM_PROVIDER=openai
# LLM_MODEL=gpt-3.5-turbo
# OPENAI_API_KEY=your-key-here

# For LM Studio
# LLM_PROVIDER=lmstudio
# LLM_MODEL=local-model

# Memory Settings
MEMORY_BACKEND=simple

# Hardware
ENABLE_HARDWARE=false

# Audio Settings
TTS_ENABLED=true
STT_ENABLED=true
TTS_PROVIDER=pyttsx3
STT_PROVIDER=whisper
""")
        logger.info("Created default .env file")


# Run on import to ensure .env exists
ensure_env_file()