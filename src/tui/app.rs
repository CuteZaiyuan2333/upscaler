// TUI 主事件循环与按键处理。
// 交互模式（vim 风格）：
//   Normal:  ↑↓←→ Tab 导航，/ 进入命令，Enter 编辑/启动
//   Command: 底部栏 / 前缀，Esc 取消，Enter 执行
//   EditingField: 底部栏显示字段名，Esc 取消，Enter 确认

use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, atomic::AtomicBool};
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::DefaultTerminal;

use super::state::{
    AppState, InputMode, RunStatus, SettingsField, Tab, TestField, TestingMode,
    TrainField,
};
use super::ui;
use super::workers::{self, TestingEvent};
use crate::training::{TrainingConfig, TrainingEvent};

const MAX_LOG_LINES: usize = 500;

pub struct App {
    pub state: AppState,
    pub training_rx: Option<Receiver<TrainingEvent>>,
    pub testing_rx: Option<Receiver<TestingEvent>>,
    terminal: DefaultTerminal,
    cancel_flag: Option<Arc<AtomicBool>>,
}

impl App {
    pub fn new(state: AppState, terminal: DefaultTerminal) -> Self {
        Self { state, training_rx: None, testing_rx: None, terminal, cancel_flag: None }
    }

    pub fn run(&mut self) -> Result<(), String> {
        let tick_rate = Duration::from_millis(50);
        while !self.state.should_quit {
            self.terminal
                .draw(|frame| ui::draw(frame, &self.state))
                .map_err(|e| e.to_string())?;

            if event::poll(tick_rate).map_err(|e| e.to_string())? {
                if let Event::Key(key) = event::read().map_err(|e| e.to_string())? {
                    if key.kind == KeyEventKind::Press {
                        self.handle_key(key);
                    }
                }
            }

            self.poll_training_events();
            self.poll_testing_events();
        }
        Ok(())
    }

    // ── 顶层按键分发 ──

    fn handle_key(&mut self, key: event::KeyEvent) {
        match self.state.input_mode {
            InputMode::Normal => self.handle_normal_key(key),
            InputMode::Command => self.handle_command_key(key),
            InputMode::EditingField => self.handle_editing_key(key),
        }
    }

    // ── Normal 模式 ──

    fn handle_normal_key(&mut self, key: event::KeyEvent) {
        match key.code {
            KeyCode::Char('/') => {
                self.state.input_mode = InputMode::Command;
                self.state.input_buffer.clear();
            }
            KeyCode::Tab => {
                self.state.active_tab = match self.state.active_tab {
                    Tab::Training => Tab::Testing,
                    Tab::Testing => Tab::Settings,
                    Tab::Settings => Tab::Training,
                };
            }
            KeyCode::Esc => {} // no-op in normal mode
            _ => match self.state.active_tab {
                Tab::Training => self.handle_training_normal(key),
                Tab::Testing => self.handle_testing_normal(key),
                Tab::Settings => self.handle_settings_normal(key),
            },
        }
    }

    // ── Command 模式 ──

    fn handle_command_key(&mut self, key: event::KeyEvent) {
        match key.code {
            KeyCode::Esc => {
                self.state.input_mode = InputMode::Normal;
                self.state.input_buffer.clear();
            }
            KeyCode::Enter => {
                let cmd = self.state.input_buffer.trim().to_lowercase();
                self.state.input_mode = InputMode::Normal;
                self.state.input_buffer.clear();
                self.execute_command(&cmd);
            }
            KeyCode::Backspace => {
                self.state.input_buffer.pop();
            }
            KeyCode::Char(c) => {
                self.state.input_buffer.push(c);
            }
            _ => {}
        }
    }

    fn execute_command(&mut self, cmd: &str) {
        match cmd {
            "quit" | "q" => self.state.should_quit = true,
            "train" => self.state.active_tab = Tab::Training,
            "test" => self.state.active_tab = Tab::Testing,
            "settings" => self.state.active_tab = Tab::Settings,
            _ => {}
        }
    }

    // ── EditingField 模式 ──

    fn handle_editing_key(&mut self, key: event::KeyEvent) {
        match key.code {
            KeyCode::Esc => {
                self.state.input_mode = InputMode::Normal;
                self.state.input_buffer.clear();
            }
            KeyCode::Enter => {
                // 确认：写回字段
                let value = self.state.input_buffer.clone();
                self.state.input_mode = InputMode::Normal;
                self.state.input_buffer.clear();
                match self.state.active_tab {
                    Tab::Training => self.write_training_field(value),
                    Tab::Testing => self.write_testing_field(value),
                    Tab::Settings => {} // 设置页无文本字段
                }
            }
            KeyCode::Backspace => {
                self.state.input_buffer.pop();
            }
            KeyCode::Char(c) => {
                self.state.input_buffer.push(c);
            }
            _ => {}
        }
    }

    // ── 训练 Normal 按键 ──

    fn handle_training_normal(&mut self, key: event::KeyEvent) {
        let t = &mut self.state.training;
        if t.status == RunStatus::Running { return; }

        match key.code {
            KeyCode::Up => t.focused_field = train_field_prev(t.focused_field),
            KeyCode::Down => t.focused_field = train_field_next(t.focused_field),
            KeyCode::Enter => match t.focused_field {
                TrainField::StartButton => self.start_training(),
                TrainField::FullResolution => t.full_resolution = !t.full_resolution,
                TrainField::CameraFilters => t.use_camera_filters = !t.use_camera_filters,
                _ => {
                    let label = train_field_label(t.focused_field);
                    self.state.editing_field_label = label.to_string();
                    self.state.input_buffer = train_field_value(t, t.focused_field).to_string();
                    self.state.input_mode = InputMode::EditingField;
                }
            },
            KeyCode::Char(' ') => match t.focused_field {
                TrainField::FullResolution => t.full_resolution = !t.full_resolution,
                TrainField::CameraFilters => t.use_camera_filters = !t.use_camera_filters,
                _ => {}
            },
            _ => {}
        }
    }

    fn write_training_field(&mut self, value: String) {
        let t = &mut self.state.training;
        match t.focused_field {
            TrainField::CheckpointDir => t.checkpoint_dir = value,
            TrainField::ResumeCheckpoint => t.resume_checkpoint = value,
            TrainField::CropSize => t.crop_size = value,
            TrainField::DatasetDir => t.dataset_dir = value,
            TrainField::StepsPerEpoch => t.steps_per_epoch = value,
            TrainField::Epochs => t.num_epochs = value,
            TrainField::LearningRate => t.learning_rate = value,
            TrainField::WeightDecay => t.weight_decay = value,
            TrainField::RefNoise => t.ref_noise = value,
            TrainField::Seed => t.seed = value,
            _ => {}
        }
    }

    // ── 测试 Normal 按键 ──

    fn handle_testing_normal(&mut self, key: event::KeyEvent) {
        let t = &mut self.state.testing;
        if t.status == RunStatus::Running { return; }

        match key.code {
            KeyCode::Up => t.focused_field = test_field_prev(t.focused_field, t.mode),
            KeyCode::Down => t.focused_field = test_field_next(t.focused_field, t.mode),
            KeyCode::Left | KeyCode::Right => {
                if t.focused_field == TestField::ModeSwitch {
                    t.mode = match t.mode {
                        TestingMode::AutoTest => TestingMode::DirectInference,
                        TestingMode::DirectInference => TestingMode::AutoTest,
                    };
                }
            }
            KeyCode::Enter => match t.focused_field {
                TestField::StartButton => self.start_testing(),
                TestField::ModeSwitch => {
                    t.mode = match t.mode {
                        TestingMode::AutoTest => TestingMode::DirectInference,
                        TestingMode::DirectInference => TestingMode::AutoTest,
                    };
                }
                _ => {
                    let label = test_field_label(t.focused_field);
                    self.state.editing_field_label = label.to_string();
                    self.state.input_buffer = test_field_value(t, t.focused_field).to_string();
                    self.state.input_mode = InputMode::EditingField;
                }
            },
            KeyCode::Char(' ') => {
                if t.focused_field == TestField::ModeSwitch {
                    t.mode = match t.mode {
                        TestingMode::AutoTest => TestingMode::DirectInference,
                        TestingMode::DirectInference => TestingMode::AutoTest,
                    };
                }
            }
            _ => {}
        }
    }

    fn write_testing_field(&mut self, value: String) {
        let t = &mut self.state.testing;
        match t.focused_field {
            TestField::TestImage => t.test_image = value,
            TestField::CheckpointPath => t.checkpoint_path = value,
            TestField::OutputDir => t.output_dir = value,
            TestField::MainImage => t.main_image = value,
            TestField::ReferenceImage => t.reference_image = value,
            TestField::OutputPath => t.output_path = value,
            _ => {}
        }
    }

    // ── 设置 Normal 按键 ──

    fn handle_settings_normal(&mut self, key: event::KeyEvent) {
        let s = &mut self.state.settings;
        match key.code {
            KeyCode::Up => s.focused_field = settings_field_prev(s.focused_field),
            KeyCode::Down => s.focused_field = settings_field_next(s.focused_field),
            KeyCode::Left => {
                match s.focused_field {
                    SettingsField::TrainingBackend => s.training_backend = s.training_backend.prev(),
                    SettingsField::InferenceBackend => s.inference_backend = s.inference_backend.prev(),
                }
            }
            KeyCode::Right => {
                match s.focused_field {
                    SettingsField::TrainingBackend => s.training_backend = s.training_backend.next(),
                    SettingsField::InferenceBackend => s.inference_backend = s.inference_backend.next(),
                }
            }
            _ => {}
        }
    }

    fn start_training(&mut self) {
        let t = &self.state.training;

        let crop_main_size: u32 = t.crop_size.parse().unwrap_or(256);
        let steps_per_epoch: usize = t.steps_per_epoch.parse().unwrap_or(1000);
        let num_epochs: usize = t.num_epochs.parse().unwrap_or(100);
        let learning_rate: f64 = t.learning_rate.parse().unwrap_or(0.0001);
        let weight_decay: f32 = t.weight_decay.parse().unwrap_or(0.0001);
        let ref_noise_std: f32 = t.ref_noise.parse().unwrap_or(0.02);
        let seed: u64 = t.seed.parse().unwrap_or(42);
        let checkpoint_every_steps: usize = t.checkpoint_interval.parse().unwrap_or(500);

        let (progress_tx, progress_rx) = mpsc::channel::<TrainingEvent>();
        let cancel_flag = Arc::new(AtomicBool::new(false));

        let config = TrainingConfig {
            num_epochs,
            steps_per_epoch,
            learning_rate,
            weight_decay,
            crop_main_size,
            checkpoint_every_steps,
            checkpoint_dir: PathBuf::from(&t.checkpoint_dir),
            dataset_dir: PathBuf::from(&t.dataset_dir),
            resume_checkpoint: PathBuf::from(&t.resume_checkpoint),
            seed,
            full_resolution: t.full_resolution,
            ref_noise_std,
            use_camera_filters: t.use_camera_filters,
            progress_sender: Some(progress_tx),
            cancel_flag: Some(cancel_flag.clone()),
        };

        self.training_rx = Some(progress_rx);
        self.cancel_flag = Some(cancel_flag);

        let t = &mut self.state.training;
        t.status = RunStatus::Running;
        t.log.clear();
        t.log.push("Training started...".into());
        t.current_epoch = 0;
        t.total_epochs = num_epochs;
        t.global_step = 0;
        t.current_loss = 0.0;
        t.avg_loss = 0.0;
        t.samples_seen = 0;

        workers::spawn_training(config, self.cancel_flag.as_ref().unwrap().clone());
    }

    fn poll_training_events(&mut self) {
        if let Some(ref rx) = self.training_rx {
            while let Ok(event) = rx.try_recv() {
                let t = &mut self.state.training;
                match event {
                    TrainingEvent::Started { dataset_size, num_params } => {
                        t.dataset_size = dataset_size;
                        t.num_params = num_params;
                        t.log.push(format!("Dataset: {} images, Params: {}", dataset_size, num_params));
                    }
                    TrainingEvent::EpochStarted { epoch, total_epochs } => {
                        t.current_epoch = epoch;
                        t.total_epochs = total_epochs;
                        t.log.push(format!("Epoch {}/{} started", epoch, total_epochs));
                    }
                    TrainingEvent::StepCompleted { epoch, global_step, loss, avg_loss, samples } => {
                        t.current_epoch = epoch;
                        t.global_step = global_step;
                        t.current_loss = loss;
                        t.avg_loss = avg_loss;
                        t.samples_seen = samples;
                        t.log.push(format!(
                            "[Epoch {:>4} | Step {:>6}] loss={:.6} avg={:.6} samples={}",
                            epoch, global_step, loss, avg_loss, samples,
                        ));
                        trim_log(&mut t.log);
                    }
                    TrainingEvent::EpochCompleted { epoch, avg_loss } => {
                        t.current_epoch = epoch;
                        t.avg_loss = avg_loss;
                        t.log.push(format!("Epoch {} completed, avg_loss={:.6}", epoch, avg_loss));
                        trim_log(&mut t.log);
                    }
                    TrainingEvent::CheckpointSaved { step } => {
                        t.log.push(format!("Checkpoint saved: step={}", step));
                        trim_log(&mut t.log);
                    }
                    TrainingEvent::Completed { total_steps } => {
                        t.status = RunStatus::Completed;
                        t.log.push(format!("Training completed! Total steps: {}", total_steps));
                        trim_log(&mut t.log);
                    }
                    TrainingEvent::Error(msg) => {
                        t.log.push(format!("ERROR: {}", msg));
                        trim_log(&mut t.log);
                    }
                }
            }
        }
    }

    // ── 测试启动 ──

    fn start_testing(&mut self) {
        let t = &mut self.state.testing;
        let (progress_tx, progress_rx) = mpsc::channel::<TestingEvent>();

        match t.mode {
            TestingMode::AutoTest => {
                if t.test_image.is_empty() {
                    t.log.push("ERROR: Test image path is required".into());
                    return;
                }
                if t.checkpoint_path.is_empty() {
                    t.log.push("ERROR: Checkpoint path is required".into());
                    return;
                }
                t.status = RunStatus::Running;
                t.log.clear();
                t.log.push("Auto test started...".into());
                t.completed = 0;
                t.total = 0;
                t.current_item = "Initializing...".into();
                workers::spawn_auto_test(
                    PathBuf::from(&t.test_image),
                    PathBuf::from(&t.checkpoint_path),
                    PathBuf::from(&t.output_dir),
                    self.state.settings.inference_backend,
                    progress_tx,
                );
            }
            TestingMode::DirectInference => {
                if t.main_image.is_empty() || t.reference_image.is_empty() {
                    t.log.push("ERROR: Main and reference image paths are required".into());
                    return;
                }
                if t.checkpoint_path.is_empty() {
                    t.log.push("ERROR: Checkpoint path is required".into());
                    return;
                }
                t.status = RunStatus::Running;
                t.log.clear();
                t.log.push("Direct inference started...".into());
                t.completed = 0;
                t.total = 1;
                t.current_item = "Running inference...".into();
                workers::spawn_direct_inference(
                    PathBuf::from(&t.main_image),
                    PathBuf::from(&t.reference_image),
                    PathBuf::from(&t.checkpoint_path),
                    PathBuf::from(&t.output_path),
                    self.state.settings.inference_backend,
                    progress_tx,
                );
            }
        }
        self.testing_rx = Some(progress_rx);
    }

    fn poll_testing_events(&mut self) {
        if let Some(ref rx) = self.testing_rx {
            while let Ok(event) = rx.try_recv() {
                let t = &mut self.state.testing;
                match event {
                    TestingEvent::AutoTestStarted { total_filters, total_checkpoints } => {
                        t.total = total_filters * total_checkpoints;
                        t.log.push(format!(
                            "Auto test: {} checkpoints × {} filters = {} total",
                            total_checkpoints, total_filters, t.total,
                        ));
                    }
                    TestingEvent::AutoTestCheckpointStarted { checkpoint } => {
                        t.current_item = format!("Checkpoint: {}", checkpoint);
                        t.log.push(format!("Testing checkpoint: {}", checkpoint));
                        trim_log(&mut t.log);
                    }
                    TestingEvent::AutoTestItemCompleted { checkpoint, filter, completed, total } => {
                        t.completed = completed;
                        t.total = total;
                        t.current_item = format!("{} / {}", checkpoint, filter);
                        if completed % 10 == 0 || completed == total {
                            t.log.push(format!("[{}/{}] {} {}", completed, total, checkpoint, filter));
                            trim_log(&mut t.log);
                        }
                    }
                    TestingEvent::AutoTestCompleted => {
                        t.status = RunStatus::Completed;
                        t.log.push("Auto test completed!".into());
                    }
                    TestingEvent::DirectInferenceStarted => {
                        t.log.push("Running inference...".into());
                    }
                    TestingEvent::DirectInferenceCompleted { output_path } => {
                        t.status = RunStatus::Completed;
                        t.completed = 1;
                        t.total = 1;
                        t.log.push(format!("Inference completed! Output: {}", output_path));
                    }
                    TestingEvent::Error(msg) => {
                        t.log.push(format!("ERROR: {}", msg));
                        trim_log(&mut t.log);
                    }
                    TestingEvent::Log(msg) => {
                        t.log.push(msg);
                        trim_log(&mut t.log);
                    }
                }
            }
        }
    }
}

// ── 字段导航辅助 ──

const TRAIN_FIELDS: &[TrainField] = &[
    TrainField::CheckpointDir,
    TrainField::ResumeCheckpoint,
    TrainField::CropSize,
    TrainField::DatasetDir,
    TrainField::StepsPerEpoch,
    TrainField::Epochs,
    TrainField::LearningRate,
    TrainField::WeightDecay,
    TrainField::RefNoise,
    TrainField::Seed,
    TrainField::FullResolution,
    TrainField::CameraFilters,
    TrainField::StartButton,
];

fn train_field_next(f: TrainField) -> TrainField {
    let pos = TRAIN_FIELDS.iter().position(|&x| x == f).unwrap_or(0);
    TRAIN_FIELDS[(pos + 1) % TRAIN_FIELDS.len()]
}

fn train_field_prev(f: TrainField) -> TrainField {
    let pos = TRAIN_FIELDS.iter().position(|&x| x == f).unwrap_or(0);
    TRAIN_FIELDS[(pos + TRAIN_FIELDS.len() - 1) % TRAIN_FIELDS.len()]
}

fn train_field_label(f: TrainField) -> &'static str {
    match f {
        TrainField::CheckpointDir => "Checkpoint Dir",
        TrainField::ResumeCheckpoint => "Resume Ckpt",
        TrainField::CropSize => "Crop Size",
        TrainField::DatasetDir => "Dataset Dir",
        TrainField::StepsPerEpoch => "Steps/Epoch",
        TrainField::Epochs => "Epochs",
        TrainField::LearningRate => "Learning Rate",
        TrainField::WeightDecay => "Weight Decay",
        TrainField::RefNoise => "Ref Noise",
        TrainField::Seed => "Seed",
        _ => "",
    }
}

fn train_field_value(t: &super::state::TrainingTabState, f: TrainField) -> &str {
    match f {
        TrainField::CheckpointDir => &t.checkpoint_dir,
        TrainField::ResumeCheckpoint => &t.resume_checkpoint,
        TrainField::CropSize => &t.crop_size,
        TrainField::DatasetDir => &t.dataset_dir,
        TrainField::StepsPerEpoch => &t.steps_per_epoch,
        TrainField::Epochs => &t.num_epochs,
        TrainField::LearningRate => &t.learning_rate,
        TrainField::WeightDecay => &t.weight_decay,
        TrainField::RefNoise => &t.ref_noise,
        TrainField::Seed => &t.seed,
        _ => "",
    }
}

const AUTO_TEST_FIELDS: &[TestField] = &[
    TestField::ModeSwitch,
    TestField::TestImage,
    TestField::CheckpointPath,
    TestField::OutputDir,
    TestField::StartButton,
];

const DIRECT_INFERENCE_FIELDS: &[TestField] = &[
    TestField::ModeSwitch,
    TestField::MainImage,
    TestField::ReferenceImage,
    TestField::CheckpointPath,
    TestField::OutputPath,
    TestField::StartButton,
];

fn test_field_prev(f: TestField, mode: TestingMode) -> TestField {
    let fields = match mode {
        TestingMode::AutoTest => AUTO_TEST_FIELDS,
        TestingMode::DirectInference => DIRECT_INFERENCE_FIELDS,
    };
    let pos = fields.iter().position(|&x| x == f).unwrap_or(0);
    fields[(pos + fields.len() - 1) % fields.len()]
}

fn test_field_next(f: TestField, mode: TestingMode) -> TestField {
    let fields = match mode {
        TestingMode::AutoTest => AUTO_TEST_FIELDS,
        TestingMode::DirectInference => DIRECT_INFERENCE_FIELDS,
    };
    let pos = fields.iter().position(|&x| x == f).unwrap_or(0);
    fields[(pos + 1) % fields.len()]
}

fn test_field_label(f: TestField) -> &'static str {
    match f {
        TestField::TestImage => "Test Image",
        TestField::CheckpointPath => "Checkpoint",
        TestField::OutputDir => "Output Dir",
        TestField::MainImage => "Main Image",
        TestField::ReferenceImage => "Reference",
        TestField::OutputPath => "Output Path",
        _ => "",
    }
}

fn test_field_value(t: &super::state::TestingTabState, f: TestField) -> &str {
    match f {
        TestField::TestImage => &t.test_image,
        TestField::CheckpointPath => &t.checkpoint_path,
        TestField::OutputDir => &t.output_dir,
        TestField::MainImage => &t.main_image,
        TestField::ReferenceImage => &t.reference_image,
        TestField::OutputPath => &t.output_path,
        _ => "",
    }
}

const SETTINGS_FIELDS: &[SettingsField] = &[
    SettingsField::TrainingBackend,
    SettingsField::InferenceBackend,
];

fn settings_field_prev(f: SettingsField) -> SettingsField {
    let pos = SETTINGS_FIELDS.iter().position(|&x| x == f).unwrap_or(0);
    SETTINGS_FIELDS[(pos + SETTINGS_FIELDS.len() - 1) % SETTINGS_FIELDS.len()]
}

fn settings_field_next(f: SettingsField) -> SettingsField {
    let pos = SETTINGS_FIELDS.iter().position(|&x| x == f).unwrap_or(0);
    SETTINGS_FIELDS[(pos + 1) % SETTINGS_FIELDS.len()]
}

fn trim_log(log: &mut Vec<String>) {
    while log.len() > MAX_LOG_LINES {
        log.remove(0);
    }
}