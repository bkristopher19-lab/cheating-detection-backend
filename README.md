# Proctoring Flutter App (Scaffold Instructions)

This folder will host the Flutter multi-platform application. If the Flutter CLI did not create it, follow these steps.

## Prerequisites
- Flutter SDK (3.22+)
- Android SDK / Xcode as applicable
- Desktop toolchains for Windows/macOS

## Create the Flutter project
```
cd proctoring
flutter create --platforms=android,ios,windows,macos app
```

## After creation
Move stubs into the generated project:
- On Windows (PowerShell): `robocopy stubs app/lib /E`
- On macOS/Linux: `cp -R stubs/* app/lib/`

## Run
```
cd app
flutter run -d windows   # or macos, android, ios
```



