use crate::plugin::Plugin;

/// 编译期注册的插件工厂；Profile 行的 `id` 对应这里的 `id`。
pub struct PluginFactory {
    pub id: &'static str,
    pub build: fn() -> Box<dyn Plugin>,
}

inventory::collect!(PluginFactory);

pub fn find_factory(id: &str) -> Option<&'static PluginFactory> {
    inventory::iter::<PluginFactory>().find(|f| f.id == id)
}

pub fn factory_ids() -> Vec<&'static str> {
    let mut v: Vec<_> = inventory::iter::<PluginFactory>().map(|f| f.id).collect();
    v.sort();
    v
}
