"""
Robot Agent Pro - Feature Test Checklist
=========================================

This script provides a comprehensive checklist to verify all features
work correctly after refactoring.

CORE FEATURES TO TEST:
======================

1. APPLICATION STARTUP
   ☐ App starts without errors
   ☐ All managers initialize (LLM, Memory, Audio, Vision)
   ☐ Configuration loads correctly
   ☐ Pages are accessible (/, /vision, /settings)

2. CHAT PAGE (/)
   ☐ Chat interface renders
   ☐ Can send text messages
   ☐ LLM responses appear correctly
   ☐ Chat bubbles display properly (user vs bot styling)
   ☐ Typing indicator shows during response
   ☐ Memory integration works (context retention)
   ☐ Mode selector works (Text vs Live)

3. VOICE FEATURES
   ☐ Voice recording button works
   ☐ Speech-to-text transcription works
   ☐ Text-to-speech output works
   ☐ Always-listening mode (VAD) works
   ☐ Noise recalibration works
   ☐ Voice cloning (Coqui) works if configured
   ☐ Multiple TTS providers switchable (pyttsx3, OpenAI, Coqui)
   ☐ Multiple STT providers switchable (Whisper, OpenAI)

4. VISION PAGE (/vision)
   ☐ Camera feed displays
   ☐ Camera can be toggled on/off
   ☐ Face detection works
   ☐ Face recognition works (if available)
   ☐ Face registration works
   ☐ Vision analysis works
   ☐ Memory search works
   ☐ Camera test function works
   ☐ Camera reset works

5. SETTINGS PAGE (/settings)
   ☐ LLM tab displays and works
      ☐ Provider selection (Anthropic/OpenAI/Ollama)
      ☐ Model selection
      ☐ API key input
      ☐ Base URL input
   ☐ Memory tab displays and works
      ☐ Backend selection (faiss/cognee)
      ☐ Max entries setting
   ☐ Voice Lab tab displays and works
      ☐ TTS provider selection
      ☐ TTS voice selection
      ☐ STT provider selection
      ☐ Voice clone recording
      ☐ Voice clone testing
      ☐ VAD threshold adjustment
   ☐ Vision tab displays and works
      ☐ Camera selection
      ☐ FPS setting
      ☐ Model selection
      ☐ Autostart toggle
      ☐ Camera test
      ☐ Camera reset
   ☐ Status tab displays
      ☐ System info shows
      ☐ Component status shows
      ☐ Uptime displays
   ☐ Save Settings button works
   ☐ Export Settings works
   ☐ Import Settings works
   ☐ Reset to Defaults works

6. CONNECTION MONITORING
   ☐ Connection status indicator shows
   ☐ Reconnection works after network issues
   ☐ Ping keeps connection alive
   ☐ Cleanup on disconnect works

7. STATE MANAGEMENT
   ☐ AppState initializes correctly
   ☐ Lazy audio initialization works
   ☐ LLM reload works
   ☐ Memory reload works
   ☐ Audio reload works
   ☐ Vision reload works
   ☐ Settings changes propagate correctly

8. ERROR HANDLING
   ☐ Missing API keys handled gracefully
   ☐ Network errors handled
   ☐ Camera errors handled
   ☐ Audio device errors handled
   ☐ Invalid settings handled

9. UI/UX
   ☐ Dark theme applies correctly
   ☐ Animations work
   ☐ Scrolling works
   ☐ Responsive layout works
   ☐ Notifications display correctly
   ☐ Loading indicators show
   ☐ Status pills update correctly

10. DATA PERSISTENCE
    ☐ Configuration persists across restarts
    ☐ Face encodings persist
    ☐ Voice samples persist
    ☐ Memory persists
    ☐ Config backups work

TESTING PROCEDURE:
==================

AUTOMATED CHECKS (run this script):
```bash
python test_features.py
```

MANUAL CHECKS:
1. Start the app: python app.py
2. Navigate to each page and verify UI renders
3. Try each feature listed above
4. Monitor console for errors
5. Check logs for warnings

QUICK SMOKE TEST:
1. Send a chat message ✓
2. Record voice message ✓
3. Check camera feed ✓
4. Change a setting and save ✓
5. Verify setting persisted ✓

"""

import sys
import os
import importlib
from pathlib import Path

# Add parent directory to path so we can import from managers
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported"""
    print("\n" + "="*50)
    print("TESTING MODULE IMPORTS")
    print("="*50)
    
    modules = [
        'managers.settings_manager',
        'managers.llm_manager',
        'managers.memory_manager',
        'managers.audio_manager',
        'managers.vision_manager',
        'managers.conversational_audio',
        'core.state',
        'core.connection',
        'pages.chat',
        'pages.vision',
        'pages.settings',
        'components.chat_bubbles',
        'components.voice_controls',
        'components.face_panel',
        'components.settings_controls',
        'utils.css',
        'utils.helpers'
    ]
    
    results = []
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
            results.append((module, True, None))
        except Exception as e:
            print(f"✗ {module}: {e}")
            results.append((module, False, str(e)))
    
    return results

def test_file_structure():
    """Test that all required files exist"""
    print("\n" + "="*50)
    print("TESTING FILE STRUCTURE")
    print("="*50)
    
    required_files = [
        'app.py',
        'core/__init__.py',
        'core/state.py',
        'core/connection.py',
        'pages/__init__.py',
        'pages/chat.py',
        'pages/vision.py',
        'pages/settings.py',
        'components/__init__.py',
        'components/chat_bubbles.py',
        'components/voice_controls.py',
        'components/face_panel.py',
        'components/settings_controls.py',
        'utils/__init__.py',
        'utils/css.py',
        'utils/helpers.py',
        'managers/__init__.py',
        'managers/settings_manager.py',
        'managers/llm_manager.py',
        'managers/memory_manager.py',
        'managers/audio_manager.py',
        'managers/vision_manager.py',
        'managers/conversational_audio.py',
        'scripts/__init__.py',
        'config.json',
        'requirements.txt'
    ]
    
    results = []
    for file in required_files:
        exists = Path(file).exists()
        symbol = "✓" if exists else "✗"
        print(f"{symbol} {file}")
        results.append((file, exists))
    
    return results

def test_config():
    """Test that config can be loaded"""
    print("\n" + "="*50)
    print("TESTING CONFIGURATION")
    print("="*50)
    
    try:
        from managers.settings_manager import config, load_settings
        config = load_settings()
        
        checks = [
            ('LLM_PROVIDER', hasattr(config, 'LLM_PROVIDER')),
            ('LLM_MODEL', hasattr(config, 'LLM_MODEL')),
            ('MEMORY_BACKEND', hasattr(config, 'MEMORY_BACKEND')),
            ('TTS_PROVIDER', hasattr(config, 'TTS_PROVIDER')),
            ('STT_PROVIDER', hasattr(config, 'STT_PROVIDER')),
        ]
        
        for name, result in checks:
            symbol = "✓" if result else "✗"
            value = getattr(config, name, 'N/A') if result else 'Missing'
            print(f"{symbol} {name}: {value}")
        
        return True, checks
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return False, str(e)

def generate_report(import_results, file_results, config_result):
    """Generate test report"""
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    import_pass = sum(1 for _, success, _ in import_results if success)
    import_total = len(import_results)
    
    file_pass = sum(1 for _, exists in file_results if exists)
    file_total = len(file_results)
    
    config_pass = config_result[0]
    
    print(f"\nImports: {import_pass}/{import_total} passed")
    print(f"Files: {file_pass}/{file_total} found")
    print(f"Config: {'✓' if config_pass else '✗'}")
    
    if import_pass == import_total and file_pass == file_total and config_pass:
        print("\n✓ ALL AUTOMATED TESTS PASSED!")
        print("\nNext steps:")
        print("1. Run: python app.py")
        print("2. Test manually using the checklist above")
        return True
    else:
        print("\n✗ SOME TESTS FAILED")
        print("\nFailed imports:")
        for module, success, error in import_results:
            if not success:
                print(f"  - {module}: {error}")
        
        print("\nMissing files:")
        for file, exists in file_results:
            if not exists:
                print(f"  - {file}")
        
        return False

if __name__ == '__main__':
    print("""
╔═══════════════════════════════════════════════════════╗
║  Robot Agent Pro - Feature Test Suite                ║
║  Verifying refactored structure...                   ║
╚═══════════════════════════════════════════════════════╝
    """)
    
    # Run tests
    import_results = test_imports()
    file_results = test_file_structure()
    config_result = test_config()
    
    # Generate report
    success = generate_report(import_results, file_results, config_result)
    
    sys.exit(0 if success else 1)
