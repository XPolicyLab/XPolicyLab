use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use oeh_proto::ServiceName;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::plugin::PluginError;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FiberId(pub u32);

struct Provided {
    value: Arc<dyn Any + Send + Sync>,
    provider: FiberId,
}

/// 按服务名提供 / 获取的注册表。类型由调用方 downcast 指定。
#[derive(Default)]
pub struct ServiceRegistry {
    inner: Mutex<HashMap<ServiceName, Provided>>,
}

impl ServiceRegistry {
    pub fn provide(
        &self,
        name: ServiceName,
        provider: FiberId,
        value: Arc<dyn Any + Send + Sync>,
    ) -> Result<(), PluginError> {
        let mut g = self.inner.lock();
        if g.contains_key(&name) {
            return Err(PluginError::AlreadyProvided(name.0));
        }
        g.insert(name, Provided { value, provider });
        Ok(())
    }

    /// 只有原 provider 才能撤销。返回是否真的撤销了。
    pub fn revoke(&self, name: &ServiceName, provider: FiberId) -> bool {
        let mut g = self.inner.lock();
        match g.get(name) {
            Some(p) if p.provider == provider => {
                g.remove(name);
                true
            }
            _ => false,
        }
    }

    pub fn get<T: Any + Send + Sync>(&self, name: &ServiceName) -> Result<Arc<T>, PluginError> {
        let g = self.inner.lock();
        let p = g
            .get(name)
            .ok_or_else(|| PluginError::MissingService(name.0.clone()))?;
        p.value
            .clone()
            .downcast::<T>()
            .map_err(|_| PluginError::ServiceType(name.0.clone()))
    }

    pub fn available(&self, name: &ServiceName) -> bool {
        self.inner.lock().contains_key(name)
    }

    pub fn provider_of(&self, name: &ServiceName) -> Option<FiberId> {
        self.inner.lock().get(name).map(|p| p.provider)
    }

    pub fn names(&self) -> Vec<ServiceName> {
        let mut v: Vec<_> = self.inner.lock().keys().cloned().collect();
        v.sort();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Foo(u32);

    #[test]
    fn provide_get_revoke() {
        let r = ServiceRegistry::default();
        let n = ServiceName::from("foo");
        assert!(!r.available(&n));
        r.provide(n.clone(), FiberId(1), Arc::new(Foo(9))).unwrap();
        assert!(r.available(&n));
        assert_eq!(r.get::<Foo>(&n).unwrap().0, 9);
        assert!(matches!(
            r.get::<String>(&n),
            Err(PluginError::ServiceType(_))
        ));
        assert!(matches!(
            r.provide(n.clone(), FiberId(2), Arc::new(Foo(1))),
            Err(PluginError::AlreadyProvided(_))
        ));
        assert!(!r.revoke(&n, FiberId(2)), "wrong provider cannot revoke");
        assert!(r.revoke(&n, FiberId(1)));
        assert!(matches!(
            r.get::<Foo>(&n),
            Err(PluginError::MissingService(_))
        ));
    }
}
