// TUI 应用状态定义。

// ── 顶层枚举 ──

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Training,
    Testing,
    Settings,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestingMode {
    AutoTest,
    DirectInference,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunStatus {
    Idle,
    Running,
    Completed,
}

/// 输入模式
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputMode {
    /// 导航模式：↑↓←→ Tab 移动焦点，/ 进入命令，Enter 进入编辑
    Normal,
    /// 编辑字段值：底部输入栏显示当前内容，Enter 确认，Esc 取消
    EditingField,
    /// 命令模式：底部输入栏显示 / 前缀，Enter 执行，Esc 取消
    Command,
}

// ── 字段标识 ──

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainField {
    CheckpointDir,
    ResumeCheckpoint,
    CropSize,
    DatasetDir,
    StepsPerEpoch,
    Epochs,
    LearningRate,
    WeightDecay,
    RefNoise,
    Seed,
    FullResolution,
    CameraFilters,
    StartButton,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestField {
    ModeSwitch,
    TestImage,
    CheckpointPath,
    OutputDir,
    MainImage,
    ReferenceImage,
    OutputPath,
    StartButton,
}

/// 可用的计算后端
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    NdArray,
    Wgpu,
    Cuda,
    Vulkan,
}

impl Backend {
    pub fn label(&self) -> &'static str {
        match self {
            Backend::NdArray => "NdArray (CPU)",
            Backend::Wgpu => "Wgpu (WebGPU)",
            Backend::Cuda => "Cuda (NVIDIA)",
            Backend::Vulkan => "Vulkan",
        }
    }

    pub fn next(&self) -> Self {
        match self {
            Backend::NdArray => Backend::Wgpu,
            Backend::Wgpu => Backend::Cuda,
            Backend::Cuda => Backend::Vulkan,
            Backend::Vulkan => Backend::NdArray,
        }
    }

    pub fn prev(&self) -> Self {
        match self {
            Backend::NdArray => Backend::Vulkan,
            Backend::Wgpu => Backend::NdArray,
            Backend::Cuda => Backend::Wgpu,
            Backend::Vulkan => Backend::Cuda,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingsField {
    TrainingBackend,
    InferenceBackend,
}

// ── 选项卡状态 ──

#[derive(Debug, Clone)]
pub struct TrainingTabState {
    pub status: RunStatus,
    pub focused_field: TrainField,
    pub checkpoint_dir: String,
    pub resume_checkpoint: String,
    pub crop_size: String,
    pub dataset_dir: String,
    pub steps_per_epoch: String,
    pub num_epochs: String,
    pub learning_rate: String,
    pub weight_decay: String,
    pub ref_noise: String,
    pub seed: String,
    pub full_resolution: bool,
    pub use_camera_filters: bool,
    pub checkpoint_interval: String,
    pub current_epoch: usize,
    pub total_epochs: usize,
    pub global_step: usize,
    pub current_loss: f32,
    pub avg_loss: f32,
    pub samples_seen: usize,
    pub dataset_size: usize,
    pub num_params: usize,
    pub log: Vec<String>,
}

impl Default for TrainingTabState {
    fn default() -> Self {
        Self {
            status: RunStatus::Idle,
            focused_field: TrainField::CheckpointDir,
            checkpoint_dir: "checkpoints".into(),
            resume_checkpoint: String::new(),
            crop_size: "256".into(),
            dataset_dir: "./dataset".into(),
            steps_per_epoch: "1000".into(),
            num_epochs: "100".into(),
            learning_rate: "0.0001".into(),
            weight_decay: "0.0001".into(),
            ref_noise: "0.02".into(),
            seed: "42".into(),
            full_resolution: false,
            use_camera_filters: true,
            checkpoint_interval: "500".into(),
            current_epoch: 0,
            total_epochs: 0,
            global_step: 0,
            current_loss: 0.0,
            avg_loss: 0.0,
            samples_seen: 0,
            dataset_size: 0,
            num_params: 0,
            log: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TestingTabState {
    pub mode: TestingMode,
    pub status: RunStatus,
    pub focused_field: TestField,
    pub test_image: String,
    pub checkpoint_path: String,
    pub output_dir: String,
    pub main_image: String,
    pub reference_image: String,
    pub output_path: String,
    pub completed: usize,
    pub total: usize,
    pub current_item: String,
    pub log: Vec<String>,
}

impl Default for TestingTabState {
    fn default() -> Self {
        Self {
            mode: TestingMode::AutoTest,
            status: RunStatus::Idle,
            focused_field: TestField::ModeSwitch,
            test_image: String::new(),
            checkpoint_path: "checkpoints".into(),
            output_dir: "test_output".into(),
            main_image: String::new(),
            reference_image: String::new(),
            output_path: "output.png".into(),
            completed: 0,
            total: 0,
            current_item: String::new(),
            log: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SettingsTabState {
    pub focused_field: SettingsField,
    pub training_backend: Backend,
    pub inference_backend: Backend,
}

impl Default for SettingsTabState {
    fn default() -> Self {
        Self {
            focused_field: SettingsField::TrainingBackend,
            training_backend: Backend::Wgpu,
            inference_backend: Backend::Wgpu,
        }
    }
}

// ── 顶层状态 ──

#[derive(Debug, Clone)]
pub struct AppState {
    pub active_tab: Tab,
    pub should_quit: bool,
    pub input_mode: InputMode,
    pub input_buffer: String,
    /// 正在编辑的字段标签（用于底部栏显示）
    pub editing_field_label: String,
    pub training: TrainingTabState,
    pub testing: TestingTabState,
    pub settings: SettingsTabState,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            active_tab: Tab::Training,
            should_quit: false,
            input_mode: InputMode::Normal,
            input_buffer: String::new(),
            editing_field_label: String::new(),
            training: TrainingTabState::default(),
            testing: TestingTabState::default(),
            settings: SettingsTabState::default(),
        }
    }
}