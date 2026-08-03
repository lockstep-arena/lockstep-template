pub use root::*;

const _: () = ::planus::check_version_compatibility("planus-1.3.0");

/// The root namespace
///
/// Generated from these locations:
/// * File `examples/rust-agent/contract/dance-off.fbs`
#[no_implicit_prelude]
#[allow(clippy::needless_lifetimes)]
mod root {
    /// The namespace `dance_off`
    ///
    /// Generated from these locations:
    /// * File `examples/rust-agent/contract/dance-off.fbs`
    pub mod dance_off {
        ///  Quaternion, (x, y, z, w), w scalar.
        ///
        /// Generated from these locations:
        /// * Struct `Quat` in the file `examples/rust-agent/contract/dance-off.fbs:21`
        #[derive(
            Copy,
            Clone,
            Debug,
            PartialEq,
            PartialOrd,
            Default,
            ::serde::Serialize,
            ::serde::Deserialize,
        )]
        pub struct Quat {
            /// The field `x` in the struct `Quat`
            pub x: f32,

            /// The field `y` in the struct `Quat`
            pub y: f32,

            /// The field `z` in the struct `Quat`
            pub z: f32,

            /// The field `w` in the struct `Quat`
            pub w: f32,
        }

        /// # Safety
        /// The Planus compiler correctly calculates `ALIGNMENT` and `SIZE`.
        unsafe impl ::planus::Primitive for Quat {
            const ALIGNMENT: usize = 4;
            const SIZE: usize = 16;
        }

        #[allow(clippy::identity_op)]
        impl ::planus::WriteAsPrimitive<Quat> for Quat {
            #[inline]
            fn write<const N: usize>(&self, cursor: ::planus::Cursor<'_, N>, buffer_position: u32) {
                let (cur, cursor) = cursor.split::<4, 12>();
                self.x.write(cur, buffer_position - 0);
                let (cur, cursor) = cursor.split::<4, 8>();
                self.y.write(cur, buffer_position - 4);
                let (cur, cursor) = cursor.split::<4, 4>();
                self.z.write(cur, buffer_position - 8);
                let (cur, cursor) = cursor.split::<4, 0>();
                self.w.write(cur, buffer_position - 12);
                cursor.finish([]);
            }
        }

        impl ::planus::WriteAsOffset<Quat> for Quat {
            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Quat> {
                unsafe {
                    builder.write_with(16, 3, |buffer_position, bytes| {
                        let bytes = bytes.as_mut_ptr();

                        ::planus::WriteAsPrimitive::write(
                            self,
                            ::planus::Cursor::new(
                                &mut *(bytes as *mut [::core::mem::MaybeUninit<u8>; 16]),
                            ),
                            buffer_position,
                        );
                    });
                }
                builder.current_offset()
            }
        }

        impl ::planus::WriteAs<Quat> for Quat {
            type Prepared = Self;
            #[inline]
            fn prepare(&self, _builder: &mut ::planus::Builder) -> Self {
                *self
            }
        }

        impl ::planus::WriteAsOptional<Quat> for Quat {
            type Prepared = Self;
            #[inline]
            fn prepare(&self, _builder: &mut ::planus::Builder) -> ::core::option::Option<Self> {
                ::core::option::Option::Some(*self)
            }
        }

        /// Reference to a deserialized [Quat].
        #[derive(Copy, Clone)]
        pub struct QuatRef<'a>(::planus::ArrayWithStartOffset<'a, 16>);

        impl<'a> QuatRef<'a> {
            /// Getter for the [`x` field](Quat#structfield.x).
            pub fn x(&self) -> f32 {
                let buffer = self.0.advance_as_array::<4>(0).unwrap();

                f32::from_le_bytes(*buffer.as_array())
            }

            /// Getter for the [`y` field](Quat#structfield.y).
            pub fn y(&self) -> f32 {
                let buffer = self.0.advance_as_array::<4>(4).unwrap();

                f32::from_le_bytes(*buffer.as_array())
            }

            /// Getter for the [`z` field](Quat#structfield.z).
            pub fn z(&self) -> f32 {
                let buffer = self.0.advance_as_array::<4>(8).unwrap();

                f32::from_le_bytes(*buffer.as_array())
            }

            /// Getter for the [`w` field](Quat#structfield.w).
            pub fn w(&self) -> f32 {
                let buffer = self.0.advance_as_array::<4>(12).unwrap();

                f32::from_le_bytes(*buffer.as_array())
            }
        }

        impl<'a> ::core::fmt::Debug for QuatRef<'a> {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                let mut f = f.debug_struct("QuatRef");
                f.field("x", &self.x());
                f.field("y", &self.y());
                f.field("z", &self.z());
                f.field("w", &self.w());
                f.finish()
            }
        }

        impl<'a> ::core::convert::From<::planus::ArrayWithStartOffset<'a, 16>> for QuatRef<'a> {
            fn from(array: ::planus::ArrayWithStartOffset<'a, 16>) -> Self {
                Self(array)
            }
        }

        impl<'a> ::core::convert::From<QuatRef<'a>> for Quat {
            #[allow(unreachable_code)]
            fn from(value: QuatRef<'a>) -> Self {
                Self {
                    x: value.x(),
                    y: value.y(),
                    z: value.z(),
                    w: value.w(),
                }
            }
        }

        impl<'a, 'b> ::core::cmp::PartialEq<QuatRef<'a>> for QuatRef<'b> {
            fn eq(&self, other: &QuatRef<'_>) -> bool {
                self.x() == other.x()
                    && self.y() == other.y()
                    && self.z() == other.z()
                    && self.w() == other.w()
            }
        }

        impl<'a, 'b> ::core::cmp::PartialOrd<QuatRef<'a>> for QuatRef<'b> {
            fn partial_cmp(
                &self,
                other: &QuatRef<'_>,
            ) -> ::core::option::Option<::core::cmp::Ordering> {
                match self.x().partial_cmp(&other.x()) {
                    ::core::option::Option::Some(::core::cmp::Ordering::Equal) => (),
                    o => return o,
                }

                match self.y().partial_cmp(&other.y()) {
                    ::core::option::Option::Some(::core::cmp::Ordering::Equal) => (),
                    o => return o,
                }

                match self.z().partial_cmp(&other.z()) {
                    ::core::option::Option::Some(::core::cmp::Ordering::Equal) => (),
                    o => return o,
                }

                self.w().partial_cmp(&other.w())
            }
        }

        impl<'a> ::planus::TableRead<'a> for QuatRef<'a> {
            #[inline]
            fn from_buffer(
                buffer: ::planus::SliceWithStartOffset<'a>,
                offset: usize,
            ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
                let buffer = buffer.advance_as_array::<16>(offset)?;
                ::core::result::Result::Ok(Self(buffer))
            }
        }

        impl<'a> ::planus::VectorRead<'a> for QuatRef<'a> {
            const STRIDE: usize = 16;

            #[inline]
            unsafe fn from_buffer(
                buffer: ::planus::SliceWithStartOffset<'a>,
                offset: usize,
            ) -> Self {
                Self(unsafe { buffer.unchecked_advance_as_array(offset) })
            }
        }

        /// # Safety
        /// The planus compiler generates implementations that initialize
        /// the bytes in `write_values`.
        unsafe impl ::planus::VectorWrite<Quat> for Quat {
            const STRIDE: usize = 16;

            type Value = Quat;

            #[inline]
            fn prepare(&self, _builder: &mut ::planus::Builder) -> Self::Value {
                *self
            }

            #[inline]
            unsafe fn write_values(
                values: &[Quat],
                bytes: *mut ::core::mem::MaybeUninit<u8>,
                buffer_position: u32,
            ) {
                let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 16];
                for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                    ::planus::WriteAsPrimitive::write(
                        v,
                        ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                        buffer_position - (16 * i) as u32,
                    );
                }
            }
        }

        ///  Rotation-vector / translation triple, engine world units.
        ///
        /// Generated from these locations:
        /// * Struct `Vec3` in the file `examples/rust-agent/contract/dance-off.fbs:24`
        #[derive(
            Copy,
            Clone,
            Debug,
            PartialEq,
            PartialOrd,
            Default,
            ::serde::Serialize,
            ::serde::Deserialize,
        )]
        pub struct Vec3 {
            /// The field `x` in the struct `Vec3`
            pub x: f32,

            /// The field `y` in the struct `Vec3`
            pub y: f32,

            /// The field `z` in the struct `Vec3`
            pub z: f32,
        }

        /// # Safety
        /// The Planus compiler correctly calculates `ALIGNMENT` and `SIZE`.
        unsafe impl ::planus::Primitive for Vec3 {
            const ALIGNMENT: usize = 4;
            const SIZE: usize = 12;
        }

        #[allow(clippy::identity_op)]
        impl ::planus::WriteAsPrimitive<Vec3> for Vec3 {
            #[inline]
            fn write<const N: usize>(&self, cursor: ::planus::Cursor<'_, N>, buffer_position: u32) {
                let (cur, cursor) = cursor.split::<4, 8>();
                self.x.write(cur, buffer_position - 0);
                let (cur, cursor) = cursor.split::<4, 4>();
                self.y.write(cur, buffer_position - 4);
                let (cur, cursor) = cursor.split::<4, 0>();
                self.z.write(cur, buffer_position - 8);
                cursor.finish([]);
            }
        }

        impl ::planus::WriteAsOffset<Vec3> for Vec3 {
            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Vec3> {
                unsafe {
                    builder.write_with(12, 3, |buffer_position, bytes| {
                        let bytes = bytes.as_mut_ptr();

                        ::planus::WriteAsPrimitive::write(
                            self,
                            ::planus::Cursor::new(
                                &mut *(bytes as *mut [::core::mem::MaybeUninit<u8>; 12]),
                            ),
                            buffer_position,
                        );
                    });
                }
                builder.current_offset()
            }
        }

        impl ::planus::WriteAs<Vec3> for Vec3 {
            type Prepared = Self;
            #[inline]
            fn prepare(&self, _builder: &mut ::planus::Builder) -> Self {
                *self
            }
        }

        impl ::planus::WriteAsOptional<Vec3> for Vec3 {
            type Prepared = Self;
            #[inline]
            fn prepare(&self, _builder: &mut ::planus::Builder) -> ::core::option::Option<Self> {
                ::core::option::Option::Some(*self)
            }
        }

        /// Reference to a deserialized [Vec3].
        #[derive(Copy, Clone)]
        pub struct Vec3Ref<'a>(::planus::ArrayWithStartOffset<'a, 12>);

        impl<'a> Vec3Ref<'a> {
            /// Getter for the [`x` field](Vec3#structfield.x).
            pub fn x(&self) -> f32 {
                let buffer = self.0.advance_as_array::<4>(0).unwrap();

                f32::from_le_bytes(*buffer.as_array())
            }

            /// Getter for the [`y` field](Vec3#structfield.y).
            pub fn y(&self) -> f32 {
                let buffer = self.0.advance_as_array::<4>(4).unwrap();

                f32::from_le_bytes(*buffer.as_array())
            }

            /// Getter for the [`z` field](Vec3#structfield.z).
            pub fn z(&self) -> f32 {
                let buffer = self.0.advance_as_array::<4>(8).unwrap();

                f32::from_le_bytes(*buffer.as_array())
            }
        }

        impl<'a> ::core::fmt::Debug for Vec3Ref<'a> {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                let mut f = f.debug_struct("Vec3Ref");
                f.field("x", &self.x());
                f.field("y", &self.y());
                f.field("z", &self.z());
                f.finish()
            }
        }

        impl<'a> ::core::convert::From<::planus::ArrayWithStartOffset<'a, 12>> for Vec3Ref<'a> {
            fn from(array: ::planus::ArrayWithStartOffset<'a, 12>) -> Self {
                Self(array)
            }
        }

        impl<'a> ::core::convert::From<Vec3Ref<'a>> for Vec3 {
            #[allow(unreachable_code)]
            fn from(value: Vec3Ref<'a>) -> Self {
                Self {
                    x: value.x(),
                    y: value.y(),
                    z: value.z(),
                }
            }
        }

        impl<'a, 'b> ::core::cmp::PartialEq<Vec3Ref<'a>> for Vec3Ref<'b> {
            fn eq(&self, other: &Vec3Ref<'_>) -> bool {
                self.x() == other.x() && self.y() == other.y() && self.z() == other.z()
            }
        }

        impl<'a, 'b> ::core::cmp::PartialOrd<Vec3Ref<'a>> for Vec3Ref<'b> {
            fn partial_cmp(
                &self,
                other: &Vec3Ref<'_>,
            ) -> ::core::option::Option<::core::cmp::Ordering> {
                match self.x().partial_cmp(&other.x()) {
                    ::core::option::Option::Some(::core::cmp::Ordering::Equal) => (),
                    o => return o,
                }

                match self.y().partial_cmp(&other.y()) {
                    ::core::option::Option::Some(::core::cmp::Ordering::Equal) => (),
                    o => return o,
                }

                self.z().partial_cmp(&other.z())
            }
        }

        impl<'a> ::planus::TableRead<'a> for Vec3Ref<'a> {
            #[inline]
            fn from_buffer(
                buffer: ::planus::SliceWithStartOffset<'a>,
                offset: usize,
            ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
                let buffer = buffer.advance_as_array::<12>(offset)?;
                ::core::result::Result::Ok(Self(buffer))
            }
        }

        impl<'a> ::planus::VectorRead<'a> for Vec3Ref<'a> {
            const STRIDE: usize = 12;

            #[inline]
            unsafe fn from_buffer(
                buffer: ::planus::SliceWithStartOffset<'a>,
                offset: usize,
            ) -> Self {
                Self(unsafe { buffer.unchecked_advance_as_array(offset) })
            }
        }

        /// # Safety
        /// The planus compiler generates implementations that initialize
        /// the bytes in `write_values`.
        unsafe impl ::planus::VectorWrite<Vec3> for Vec3 {
            const STRIDE: usize = 12;

            type Value = Vec3;

            #[inline]
            fn prepare(&self, _builder: &mut ::planus::Builder) -> Self::Value {
                *self
            }

            #[inline]
            unsafe fn write_values(
                values: &[Vec3],
                bytes: *mut ::core::mem::MaybeUninit<u8>,
                buffer_position: u32,
            ) {
                let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 12];
                for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                    ::planus::WriteAsPrimitive::write(
                        v,
                        ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                        buffer_position - (12 * i) as u32,
                    );
                }
            }
        }

        ///  One upcoming card, as structured data: the routine is legible directly —
        ///  the strip image is what the board shows, not an OCR puzzle.
        ///
        /// Generated from these locations:
        /// * Struct `CardCue` in the file `examples/rust-agent/contract/dance-off.fbs:28`
        #[derive(
            Copy,
            Clone,
            Debug,
            PartialEq,
            PartialOrd,
            Eq,
            Ord,
            Hash,
            Default,
            ::serde::Serialize,
            ::serde::Deserialize,
        )]
        pub struct CardCue {
            ///  Index into the move vocabulary.
            pub move_id: u32,

            ///  Modifier id; the target pose is composed from (move_id, modifier).
            pub modifier: u32,

            ///  Ticks until this card's centre crosses the hit line. Negative once
            ///  past it while still inside its scoring window.
            pub ticks_to_hit: i32,
        }

        /// # Safety
        /// The Planus compiler correctly calculates `ALIGNMENT` and `SIZE`.
        unsafe impl ::planus::Primitive for CardCue {
            const ALIGNMENT: usize = 4;
            const SIZE: usize = 12;
        }

        #[allow(clippy::identity_op)]
        impl ::planus::WriteAsPrimitive<CardCue> for CardCue {
            #[inline]
            fn write<const N: usize>(&self, cursor: ::planus::Cursor<'_, N>, buffer_position: u32) {
                let (cur, cursor) = cursor.split::<4, 8>();
                self.move_id.write(cur, buffer_position - 0);
                let (cur, cursor) = cursor.split::<4, 4>();
                self.modifier.write(cur, buffer_position - 4);
                let (cur, cursor) = cursor.split::<4, 0>();
                self.ticks_to_hit.write(cur, buffer_position - 8);
                cursor.finish([]);
            }
        }

        impl ::planus::WriteAsOffset<CardCue> for CardCue {
            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<CardCue> {
                unsafe {
                    builder.write_with(12, 3, |buffer_position, bytes| {
                        let bytes = bytes.as_mut_ptr();

                        ::planus::WriteAsPrimitive::write(
                            self,
                            ::planus::Cursor::new(
                                &mut *(bytes as *mut [::core::mem::MaybeUninit<u8>; 12]),
                            ),
                            buffer_position,
                        );
                    });
                }
                builder.current_offset()
            }
        }

        impl ::planus::WriteAs<CardCue> for CardCue {
            type Prepared = Self;
            #[inline]
            fn prepare(&self, _builder: &mut ::planus::Builder) -> Self {
                *self
            }
        }

        impl ::planus::WriteAsOptional<CardCue> for CardCue {
            type Prepared = Self;
            #[inline]
            fn prepare(&self, _builder: &mut ::planus::Builder) -> ::core::option::Option<Self> {
                ::core::option::Option::Some(*self)
            }
        }

        /// Reference to a deserialized [CardCue].
        #[derive(Copy, Clone)]
        pub struct CardCueRef<'a>(::planus::ArrayWithStartOffset<'a, 12>);

        impl<'a> CardCueRef<'a> {
            /// Getter for the [`move_id` field](CardCue#structfield.move_id).
            pub fn move_id(&self) -> u32 {
                let buffer = self.0.advance_as_array::<4>(0).unwrap();

                u32::from_le_bytes(*buffer.as_array())
            }

            /// Getter for the [`modifier` field](CardCue#structfield.modifier).
            pub fn modifier(&self) -> u32 {
                let buffer = self.0.advance_as_array::<4>(4).unwrap();

                u32::from_le_bytes(*buffer.as_array())
            }

            /// Getter for the [`ticks_to_hit` field](CardCue#structfield.ticks_to_hit).
            pub fn ticks_to_hit(&self) -> i32 {
                let buffer = self.0.advance_as_array::<4>(8).unwrap();

                i32::from_le_bytes(*buffer.as_array())
            }
        }

        impl<'a> ::core::fmt::Debug for CardCueRef<'a> {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                let mut f = f.debug_struct("CardCueRef");
                f.field("move_id", &self.move_id());
                f.field("modifier", &self.modifier());
                f.field("ticks_to_hit", &self.ticks_to_hit());
                f.finish()
            }
        }

        impl<'a> ::core::convert::From<::planus::ArrayWithStartOffset<'a, 12>> for CardCueRef<'a> {
            fn from(array: ::planus::ArrayWithStartOffset<'a, 12>) -> Self {
                Self(array)
            }
        }

        impl<'a> ::core::convert::From<CardCueRef<'a>> for CardCue {
            #[allow(unreachable_code)]
            fn from(value: CardCueRef<'a>) -> Self {
                Self {
                    move_id: value.move_id(),
                    modifier: value.modifier(),
                    ticks_to_hit: value.ticks_to_hit(),
                }
            }
        }

        impl<'a, 'b> ::core::cmp::PartialEq<CardCueRef<'a>> for CardCueRef<'b> {
            fn eq(&self, other: &CardCueRef<'_>) -> bool {
                self.move_id() == other.move_id()
                    && self.modifier() == other.modifier()
                    && self.ticks_to_hit() == other.ticks_to_hit()
            }
        }

        impl<'a> ::core::cmp::Eq for CardCueRef<'a> {}
        impl<'a, 'b> ::core::cmp::PartialOrd<CardCueRef<'a>> for CardCueRef<'b> {
            fn partial_cmp(
                &self,
                other: &CardCueRef<'_>,
            ) -> ::core::option::Option<::core::cmp::Ordering> {
                ::core::option::Option::Some(::core::cmp::Ord::cmp(self, other))
            }
        }

        impl<'a> ::core::cmp::Ord for CardCueRef<'a> {
            fn cmp(&self, other: &CardCueRef<'_>) -> ::core::cmp::Ordering {
                self.move_id()
                    .cmp(&other.move_id())
                    .then_with(|| self.modifier().cmp(&other.modifier()))
                    .then_with(|| self.ticks_to_hit().cmp(&other.ticks_to_hit()))
            }
        }

        impl<'a> ::core::hash::Hash for CardCueRef<'a> {
            fn hash<H: ::core::hash::Hasher>(&self, state: &mut H) {
                self.move_id().hash(state);
                self.modifier().hash(state);
                self.ticks_to_hit().hash(state);
            }
        }

        impl<'a> ::planus::TableRead<'a> for CardCueRef<'a> {
            #[inline]
            fn from_buffer(
                buffer: ::planus::SliceWithStartOffset<'a>,
                offset: usize,
            ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
                let buffer = buffer.advance_as_array::<12>(offset)?;
                ::core::result::Result::Ok(Self(buffer))
            }
        }

        impl<'a> ::planus::VectorRead<'a> for CardCueRef<'a> {
            const STRIDE: usize = 12;

            #[inline]
            unsafe fn from_buffer(
                buffer: ::planus::SliceWithStartOffset<'a>,
                offset: usize,
            ) -> Self {
                Self(unsafe { buffer.unchecked_advance_as_array(offset) })
            }
        }

        /// # Safety
        /// The planus compiler generates implementations that initialize
        /// the bytes in `write_values`.
        unsafe impl ::planus::VectorWrite<CardCue> for CardCue {
            const STRIDE: usize = 12;

            type Value = CardCue;

            #[inline]
            fn prepare(&self, _builder: &mut ::planus::Builder) -> Self::Value {
                *self
            }

            #[inline]
            unsafe fn write_values(
                values: &[CardCue],
                bytes: *mut ::core::mem::MaybeUninit<u8>,
                buffer_position: u32,
            ) {
                let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 12];
                for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                    ::planus::WriteAsPrimitive::write(
                        v,
                        ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                        buffer_position - (12 * i) as u32,
                    );
                }
            }
        }

        ///  A dancer pose on the wire: pelvis (root) world transform + the 12
        ///  joint-local rotations in canonical joint order.
        ///
        /// Generated from these locations:
        /// * Table `Pose` in the file `examples/rust-agent/contract/dance-off.fbs:40`
        #[derive(Clone, Debug, PartialEq, PartialOrd, ::serde::Serialize, ::serde::Deserialize)]
        pub struct Pose {
            /// The field `root_translation` in the table `Pose`
            pub root_translation: self::Vec3,
            /// The field `root_rotation` in the table `Pose`
            pub root_rotation: self::Quat,
            ///  Canonical joint order; short/malformed lists zero-pad to identity.
            pub joints: ::planus::alloc::vec::Vec<self::Quat>,
        }

        #[allow(clippy::derivable_impls)]
        impl ::core::default::Default for Pose {
            fn default() -> Self {
                Self {
                    root_translation: ::core::default::Default::default(),
                    root_rotation: ::core::default::Default::default(),
                    joints: ::core::default::Default::default(),
                }
            }
        }

        impl Pose {
            /// Creates a [PoseBuilder] for serializing an instance of this table.
            #[inline]
            pub fn builder() -> PoseBuilder<()> {
                PoseBuilder(())
            }

            #[allow(clippy::too_many_arguments)]
            pub fn create(
                builder: &mut ::planus::Builder,
                field_root_translation: impl ::planus::WriteAs<self::Vec3>,
                field_root_rotation: impl ::planus::WriteAs<self::Quat>,
                field_joints: impl ::planus::WriteAs<::planus::Offset<[self::Quat]>>,
            ) -> ::planus::Offset<Self> {
                let prepared_root_translation = field_root_translation.prepare(builder);
                let prepared_root_rotation = field_root_rotation.prepare(builder);
                let prepared_joints = field_joints.prepare(builder);

                let mut table_writer: ::planus::table_writer::TableWriter<10> =
                    ::core::default::Default::default();
                table_writer.write_entry::<self::Vec3>(0);
                table_writer.write_entry::<self::Quat>(1);
                table_writer.write_entry::<::planus::Offset<[self::Quat]>>(2);

                unsafe {
                    table_writer.finish(builder, |object_writer| {
                        object_writer.write::<_, _, 12>(&prepared_root_translation);
                        object_writer.write::<_, _, 16>(&prepared_root_rotation);
                        object_writer.write::<_, _, 4>(&prepared_joints);
                    });
                }
                builder.current_offset()
            }
        }

        impl ::planus::WriteAs<::planus::Offset<Pose>> for Pose {
            type Prepared = ::planus::Offset<Self>;

            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Pose> {
                ::planus::WriteAsOffset::prepare(self, builder)
            }
        }

        impl ::planus::WriteAsOptional<::planus::Offset<Pose>> for Pose {
            type Prepared = ::planus::Offset<Self>;

            #[inline]
            fn prepare(
                &self,
                builder: &mut ::planus::Builder,
            ) -> ::core::option::Option<::planus::Offset<Pose>> {
                ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
            }
        }

        impl ::planus::WriteAsOffset<Pose> for Pose {
            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Pose> {
                Pose::create(
                    builder,
                    self.root_translation,
                    self.root_rotation,
                    &self.joints,
                )
            }
        }

        /// Builder for serializing an instance of the [Pose] type.
        ///
        /// Can be created using the [Pose::builder] method.
        #[derive(Debug)]
        #[must_use]
        pub struct PoseBuilder<State>(State);

        impl PoseBuilder<()> {
            /// Setter for the [`root_translation` field](Pose#structfield.root_translation).
            #[inline]
            #[allow(clippy::type_complexity)]
            pub fn root_translation<T0>(self, value: T0) -> PoseBuilder<(T0,)>
            where
                T0: ::planus::WriteAs<self::Vec3>,
            {
                PoseBuilder((value,))
            }
        }

        impl<T0> PoseBuilder<(T0,)> {
            /// Setter for the [`root_rotation` field](Pose#structfield.root_rotation).
            #[inline]
            #[allow(clippy::type_complexity)]
            pub fn root_rotation<T1>(self, value: T1) -> PoseBuilder<(T0, T1)>
            where
                T1: ::planus::WriteAs<self::Quat>,
            {
                let (v0,) = self.0;
                PoseBuilder((v0, value))
            }
        }

        impl<T0, T1> PoseBuilder<(T0, T1)> {
            /// Setter for the [`joints` field](Pose#structfield.joints).
            #[inline]
            #[allow(clippy::type_complexity)]
            pub fn joints<T2>(self, value: T2) -> PoseBuilder<(T0, T1, T2)>
            where
                T2: ::planus::WriteAs<::planus::Offset<[self::Quat]>>,
            {
                let (v0, v1) = self.0;
                PoseBuilder((v0, v1, value))
            }
        }

        impl<T0, T1, T2> PoseBuilder<(T0, T1, T2)> {
            /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Pose].
            #[inline]
            pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<Pose>
            where
                Self: ::planus::WriteAsOffset<Pose>,
            {
                ::planus::WriteAsOffset::prepare(&self, builder)
            }
        }

        impl<
                T0: ::planus::WriteAs<self::Vec3>,
                T1: ::planus::WriteAs<self::Quat>,
                T2: ::planus::WriteAs<::planus::Offset<[self::Quat]>>,
            > ::planus::WriteAs<::planus::Offset<Pose>> for PoseBuilder<(T0, T1, T2)>
        {
            type Prepared = ::planus::Offset<Pose>;

            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Pose> {
                ::planus::WriteAsOffset::prepare(self, builder)
            }
        }

        impl<
                T0: ::planus::WriteAs<self::Vec3>,
                T1: ::planus::WriteAs<self::Quat>,
                T2: ::planus::WriteAs<::planus::Offset<[self::Quat]>>,
            > ::planus::WriteAsOptional<::planus::Offset<Pose>> for PoseBuilder<(T0, T1, T2)>
        {
            type Prepared = ::planus::Offset<Pose>;

            #[inline]
            fn prepare(
                &self,
                builder: &mut ::planus::Builder,
            ) -> ::core::option::Option<::planus::Offset<Pose>> {
                ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
            }
        }

        impl<
                T0: ::planus::WriteAs<self::Vec3>,
                T1: ::planus::WriteAs<self::Quat>,
                T2: ::planus::WriteAs<::planus::Offset<[self::Quat]>>,
            > ::planus::WriteAsOffset<Pose> for PoseBuilder<(T0, T1, T2)>
        {
            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Pose> {
                let (v0, v1, v2) = &self.0;
                Pose::create(builder, v0, v1, v2)
            }
        }

        /// Reference to a deserialized [Pose].
        #[derive(Copy, Clone)]
        pub struct PoseRef<'a>(#[allow(dead_code)] ::planus::table_reader::Table<'a>);

        impl<'a> PoseRef<'a> {
            /// Getter for the [`root_translation` field](Pose#structfield.root_translation).
            #[inline]
            pub fn root_translation(&self) -> ::planus::Result<self::Vec3Ref<'a>> {
                self.0.access_required(0, "Pose", "root_translation")
            }

            /// Getter for the [`root_rotation` field](Pose#structfield.root_rotation).
            #[inline]
            pub fn root_rotation(&self) -> ::planus::Result<self::QuatRef<'a>> {
                self.0.access_required(1, "Pose", "root_rotation")
            }

            /// Getter for the [`joints` field](Pose#structfield.joints).
            #[inline]
            pub fn joints(&self) -> ::planus::Result<::planus::Vector<'a, self::QuatRef<'a>>> {
                self.0.access_required(2, "Pose", "joints")
            }
        }

        impl<'a> ::core::fmt::Debug for PoseRef<'a> {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                let mut f = f.debug_struct("PoseRef");
                f.field("root_translation", &self.root_translation());
                f.field("root_rotation", &self.root_rotation());
                f.field("joints", &self.joints());
                f.finish()
            }
        }

        impl<'a> ::core::convert::TryFrom<PoseRef<'a>> for Pose {
            type Error = ::planus::Error;

            #[allow(unreachable_code)]
            fn try_from(value: PoseRef<'a>) -> ::planus::Result<Self> {
                ::core::result::Result::Ok(Self {
                    root_translation: ::core::convert::Into::into(value.root_translation()?),
                    root_rotation: ::core::convert::Into::into(value.root_rotation()?),
                    joints: value.joints()?.to_vec()?,
                })
            }
        }

        impl<'a> ::planus::TableRead<'a> for PoseRef<'a> {
            #[inline]
            fn from_buffer(
                buffer: ::planus::SliceWithStartOffset<'a>,
                offset: usize,
            ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
                ::core::result::Result::Ok(Self(::planus::table_reader::Table::from_buffer(
                    buffer, offset,
                )?))
            }
        }

        impl<'a> ::planus::VectorReadInner<'a> for PoseRef<'a> {
            type Error = ::planus::Error;
            const STRIDE: usize = 4;

            unsafe fn from_buffer(
                buffer: ::planus::SliceWithStartOffset<'a>,
                offset: usize,
            ) -> ::planus::Result<Self> {
                ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                    error_kind.with_error_location("[PoseRef]", "get", buffer.offset_from_start)
                })
            }
        }

        /// # Safety
        /// The planus compiler generates implementations that initialize
        /// the bytes in `write_values`.
        unsafe impl ::planus::VectorWrite<::planus::Offset<Pose>> for Pose {
            type Value = ::planus::Offset<Pose>;
            const STRIDE: usize = 4;
            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                ::planus::WriteAs::prepare(self, builder)
            }

            #[inline]
            unsafe fn write_values(
                values: &[::planus::Offset<Pose>],
                bytes: *mut ::core::mem::MaybeUninit<u8>,
                buffer_position: u32,
            ) {
                let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                    ::planus::WriteAsPrimitive::write(
                        v,
                        ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                        buffer_position - (Self::STRIDE * i) as u32,
                    );
                }
            }
        }

        impl<'a> ::planus::ReadAsRoot<'a> for PoseRef<'a> {
            fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                ::planus::TableRead::from_buffer(
                    ::planus::SliceWithStartOffset {
                        buffer: slice,
                        offset_from_start: 0,
                    },
                    0,
                )
                .map_err(|error_kind| {
                    error_kind.with_error_location("[PoseRef]", "read_as_root", 0)
                })
            }
        }

        ///  Public dancer status (fall/recovery + scoring HUD fields).
        ///
        /// Generated from these locations:
        /// * Table `Status` in the file `examples/rust-agent/contract/dance-off.fbs:48`
        #[derive(Clone, Debug, PartialEq, PartialOrd, ::serde::Serialize, ::serde::Deserialize)]
        pub struct Status {
            /// The field `fallen` in the table `Status`
            pub fallen: bool,
            /// The field `down_ticks_left` in the table `Status`
            pub down_ticks_left: u32,
            /// The field `combo` in the table `Status`
            pub combo: u32,
            /// The field `last_move_points` in the table `Status`
            pub last_move_points: f32,
            /// The field `falls` in the table `Status`
            pub falls: u32,
        }

        #[allow(clippy::derivable_impls)]
        impl ::core::default::Default for Status {
            fn default() -> Self {
                Self {
                    fallen: false,
                    down_ticks_left: 0,
                    combo: 0,
                    last_move_points: 0.0,
                    falls: 0,
                }
            }
        }

        impl Status {
            /// Creates a [StatusBuilder] for serializing an instance of this table.
            #[inline]
            pub fn builder() -> StatusBuilder<()> {
                StatusBuilder(())
            }

            #[allow(clippy::too_many_arguments)]
            pub fn create(
                builder: &mut ::planus::Builder,
                field_fallen: impl ::planus::WriteAsDefault<bool, bool>,
                field_down_ticks_left: impl ::planus::WriteAsDefault<u32, u32>,
                field_combo: impl ::planus::WriteAsDefault<u32, u32>,
                field_last_move_points: impl ::planus::WriteAsDefault<f32, f32>,
                field_falls: impl ::planus::WriteAsDefault<u32, u32>,
            ) -> ::planus::Offset<Self> {
                let prepared_fallen = field_fallen.prepare(builder, &false);
                let prepared_down_ticks_left = field_down_ticks_left.prepare(builder, &0);
                let prepared_combo = field_combo.prepare(builder, &0);
                let prepared_last_move_points = field_last_move_points.prepare(builder, &0.0);
                let prepared_falls = field_falls.prepare(builder, &0);

                let mut table_writer: ::planus::table_writer::TableWriter<14> =
                    ::core::default::Default::default();
                if prepared_down_ticks_left.is_some() {
                    table_writer.write_entry::<u32>(1);
                }
                if prepared_combo.is_some() {
                    table_writer.write_entry::<u32>(2);
                }
                if prepared_last_move_points.is_some() {
                    table_writer.write_entry::<f32>(3);
                }
                if prepared_falls.is_some() {
                    table_writer.write_entry::<u32>(4);
                }
                if prepared_fallen.is_some() {
                    table_writer.write_entry::<bool>(0);
                }

                unsafe {
                    table_writer.finish(builder, |object_writer| {
                        if let ::core::option::Option::Some(prepared_down_ticks_left) =
                            prepared_down_ticks_left
                        {
                            object_writer.write::<_, _, 4>(&prepared_down_ticks_left);
                        }
                        if let ::core::option::Option::Some(prepared_combo) = prepared_combo {
                            object_writer.write::<_, _, 4>(&prepared_combo);
                        }
                        if let ::core::option::Option::Some(prepared_last_move_points) =
                            prepared_last_move_points
                        {
                            object_writer.write::<_, _, 4>(&prepared_last_move_points);
                        }
                        if let ::core::option::Option::Some(prepared_falls) = prepared_falls {
                            object_writer.write::<_, _, 4>(&prepared_falls);
                        }
                        if let ::core::option::Option::Some(prepared_fallen) = prepared_fallen {
                            object_writer.write::<_, _, 1>(&prepared_fallen);
                        }
                    });
                }
                builder.current_offset()
            }
        }

        impl ::planus::WriteAs<::planus::Offset<Status>> for Status {
            type Prepared = ::planus::Offset<Self>;

            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Status> {
                ::planus::WriteAsOffset::prepare(self, builder)
            }
        }

        impl ::planus::WriteAsOptional<::planus::Offset<Status>> for Status {
            type Prepared = ::planus::Offset<Self>;

            #[inline]
            fn prepare(
                &self,
                builder: &mut ::planus::Builder,
            ) -> ::core::option::Option<::planus::Offset<Status>> {
                ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
            }
        }

        impl ::planus::WriteAsOffset<Status> for Status {
            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Status> {
                Status::create(
                    builder,
                    self.fallen,
                    self.down_ticks_left,
                    self.combo,
                    self.last_move_points,
                    self.falls,
                )
            }
        }

        /// Builder for serializing an instance of the [Status] type.
        ///
        /// Can be created using the [Status::builder] method.
        #[derive(Debug)]
        #[must_use]
        pub struct StatusBuilder<State>(State);

        impl StatusBuilder<()> {
            /// Setter for the [`fallen` field](Status#structfield.fallen).
            #[inline]
            #[allow(clippy::type_complexity)]
            pub fn fallen<T0>(self, value: T0) -> StatusBuilder<(T0,)>
            where
                T0: ::planus::WriteAsDefault<bool, bool>,
            {
                StatusBuilder((value,))
            }

            /// Sets the [`fallen` field](Status#structfield.fallen) to the default value.
            #[inline]
            #[allow(clippy::type_complexity)]
            pub fn fallen_as_default(self) -> StatusBuilder<(::planus::DefaultValue,)> {
                self.fallen(::planus::DefaultValue)
            }
        }

        impl<T0> StatusBuilder<(T0,)> {
            /// Setter for the [`down_ticks_left` field](Status#structfield.down_ticks_left).
            #[inline]
            #[allow(clippy::type_complexity)]
            pub fn down_ticks_left<T1>(self, value: T1) -> StatusBuilder<(T0, T1)>
            where
                T1: ::planus::WriteAsDefault<u32, u32>,
            {
                let (v0,) = self.0;
                StatusBuilder((v0, value))
            }

            /// Sets the [`down_ticks_left` field](Status#structfield.down_ticks_left) to the default value.
            #[inline]
            #[allow(clippy::type_complexity)]
            pub fn down_ticks_left_as_default(self) -> StatusBuilder<(T0, ::planus::DefaultValue)> {
                self.down_ticks_left(::planus::DefaultValue)
            }
        }

        impl<T0, T1> StatusBuilder<(T0, T1)> {
            /// Setter for the [`combo` field](Status#structfield.combo).
            #[inline]
            #[allow(clippy::type_complexity)]
            pub fn combo<T2>(self, value: T2) -> StatusBuilder<(T0, T1, T2)>
            where
                T2: ::planus::WriteAsDefault<u32, u32>,
            {
                let (v0, v1) = self.0;
                StatusBuilder((v0, v1, value))
            }

            /// Sets the [`combo` field](Status#structfield.combo) to the default value.
            #[inline]
            #[allow(clippy::type_complexity)]
            pub fn combo_as_default(self) -> StatusBuilder<(T0, T1, ::planus::DefaultValue)> {
                self.combo(::planus::DefaultValue)
            }
        }

        impl<T0, T1, T2> StatusBuilder<(T0, T1, T2)> {
            /// Setter for the [`last_move_points` field](Status#structfield.last_move_points).
            #[inline]
            #[allow(clippy::type_complexity)]
            pub fn last_move_points<T3>(self, value: T3) -> StatusBuilder<(T0, T1, T2, T3)>
            where
                T3: ::planus::WriteAsDefault<f32, f32>,
            {
                let (v0, v1, v2) = self.0;
                StatusBuilder((v0, v1, v2, value))
            }

            /// Sets the [`last_move_points` field](Status#structfield.last_move_points) to the default value.
            #[inline]
            #[allow(clippy::type_complexity)]
            pub fn last_move_points_as_default(
                self,
            ) -> StatusBuilder<(T0, T1, T2, ::planus::DefaultValue)> {
                self.last_move_points(::planus::DefaultValue)
            }
        }

        impl<T0, T1, T2, T3> StatusBuilder<(T0, T1, T2, T3)> {
            /// Setter for the [`falls` field](Status#structfield.falls).
            #[inline]
            #[allow(clippy::type_complexity)]
            pub fn falls<T4>(self, value: T4) -> StatusBuilder<(T0, T1, T2, T3, T4)>
            where
                T4: ::planus::WriteAsDefault<u32, u32>,
            {
                let (v0, v1, v2, v3) = self.0;
                StatusBuilder((v0, v1, v2, v3, value))
            }

            /// Sets the [`falls` field](Status#structfield.falls) to the default value.
            #[inline]
            #[allow(clippy::type_complexity)]
            pub fn falls_as_default(
                self,
            ) -> StatusBuilder<(T0, T1, T2, T3, ::planus::DefaultValue)> {
                self.falls(::planus::DefaultValue)
            }
        }

        impl<T0, T1, T2, T3, T4> StatusBuilder<(T0, T1, T2, T3, T4)> {
            /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Status].
            #[inline]
            pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<Status>
            where
                Self: ::planus::WriteAsOffset<Status>,
            {
                ::planus::WriteAsOffset::prepare(&self, builder)
            }
        }

        impl<
                T0: ::planus::WriteAsDefault<bool, bool>,
                T1: ::planus::WriteAsDefault<u32, u32>,
                T2: ::planus::WriteAsDefault<u32, u32>,
                T3: ::planus::WriteAsDefault<f32, f32>,
                T4: ::planus::WriteAsDefault<u32, u32>,
            > ::planus::WriteAs<::planus::Offset<Status>> for StatusBuilder<(T0, T1, T2, T3, T4)>
        {
            type Prepared = ::planus::Offset<Status>;

            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Status> {
                ::planus::WriteAsOffset::prepare(self, builder)
            }
        }

        impl<
                T0: ::planus::WriteAsDefault<bool, bool>,
                T1: ::planus::WriteAsDefault<u32, u32>,
                T2: ::planus::WriteAsDefault<u32, u32>,
                T3: ::planus::WriteAsDefault<f32, f32>,
                T4: ::planus::WriteAsDefault<u32, u32>,
            > ::planus::WriteAsOptional<::planus::Offset<Status>>
            for StatusBuilder<(T0, T1, T2, T3, T4)>
        {
            type Prepared = ::planus::Offset<Status>;

            #[inline]
            fn prepare(
                &self,
                builder: &mut ::planus::Builder,
            ) -> ::core::option::Option<::planus::Offset<Status>> {
                ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
            }
        }

        impl<
                T0: ::planus::WriteAsDefault<bool, bool>,
                T1: ::planus::WriteAsDefault<u32, u32>,
                T2: ::planus::WriteAsDefault<u32, u32>,
                T3: ::planus::WriteAsDefault<f32, f32>,
                T4: ::planus::WriteAsDefault<u32, u32>,
            > ::planus::WriteAsOffset<Status> for StatusBuilder<(T0, T1, T2, T3, T4)>
        {
            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Status> {
                let (v0, v1, v2, v3, v4) = &self.0;
                Status::create(builder, v0, v1, v2, v3, v4)
            }
        }

        /// Reference to a deserialized [Status].
        #[derive(Copy, Clone)]
        pub struct StatusRef<'a>(#[allow(dead_code)] ::planus::table_reader::Table<'a>);

        impl<'a> StatusRef<'a> {
            /// Getter for the [`fallen` field](Status#structfield.fallen).
            #[inline]
            pub fn fallen(&self) -> ::planus::Result<bool> {
                ::core::result::Result::Ok(self.0.access(0, "Status", "fallen")?.unwrap_or(false))
            }

            /// Getter for the [`down_ticks_left` field](Status#structfield.down_ticks_left).
            #[inline]
            pub fn down_ticks_left(&self) -> ::planus::Result<u32> {
                ::core::result::Result::Ok(
                    self.0.access(1, "Status", "down_ticks_left")?.unwrap_or(0),
                )
            }

            /// Getter for the [`combo` field](Status#structfield.combo).
            #[inline]
            pub fn combo(&self) -> ::planus::Result<u32> {
                ::core::result::Result::Ok(self.0.access(2, "Status", "combo")?.unwrap_or(0))
            }

            /// Getter for the [`last_move_points` field](Status#structfield.last_move_points).
            #[inline]
            pub fn last_move_points(&self) -> ::planus::Result<f32> {
                ::core::result::Result::Ok(
                    self.0
                        .access(3, "Status", "last_move_points")?
                        .unwrap_or(0.0),
                )
            }

            /// Getter for the [`falls` field](Status#structfield.falls).
            #[inline]
            pub fn falls(&self) -> ::planus::Result<u32> {
                ::core::result::Result::Ok(self.0.access(4, "Status", "falls")?.unwrap_or(0))
            }
        }

        impl<'a> ::core::fmt::Debug for StatusRef<'a> {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                let mut f = f.debug_struct("StatusRef");
                f.field("fallen", &self.fallen());
                f.field("down_ticks_left", &self.down_ticks_left());
                f.field("combo", &self.combo());
                f.field("last_move_points", &self.last_move_points());
                f.field("falls", &self.falls());
                f.finish()
            }
        }

        impl<'a> ::core::convert::TryFrom<StatusRef<'a>> for Status {
            type Error = ::planus::Error;

            #[allow(unreachable_code)]
            fn try_from(value: StatusRef<'a>) -> ::planus::Result<Self> {
                ::core::result::Result::Ok(Self {
                    fallen: ::core::convert::TryInto::try_into(value.fallen()?)?,
                    down_ticks_left: ::core::convert::TryInto::try_into(value.down_ticks_left()?)?,
                    combo: ::core::convert::TryInto::try_into(value.combo()?)?,
                    last_move_points: ::core::convert::TryInto::try_into(
                        value.last_move_points()?,
                    )?,
                    falls: ::core::convert::TryInto::try_into(value.falls()?)?,
                })
            }
        }

        impl<'a> ::planus::TableRead<'a> for StatusRef<'a> {
            #[inline]
            fn from_buffer(
                buffer: ::planus::SliceWithStartOffset<'a>,
                offset: usize,
            ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
                ::core::result::Result::Ok(Self(::planus::table_reader::Table::from_buffer(
                    buffer, offset,
                )?))
            }
        }

        impl<'a> ::planus::VectorReadInner<'a> for StatusRef<'a> {
            type Error = ::planus::Error;
            const STRIDE: usize = 4;

            unsafe fn from_buffer(
                buffer: ::planus::SliceWithStartOffset<'a>,
                offset: usize,
            ) -> ::planus::Result<Self> {
                ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                    error_kind.with_error_location("[StatusRef]", "get", buffer.offset_from_start)
                })
            }
        }

        /// # Safety
        /// The planus compiler generates implementations that initialize
        /// the bytes in `write_values`.
        unsafe impl ::planus::VectorWrite<::planus::Offset<Status>> for Status {
            type Value = ::planus::Offset<Status>;
            const STRIDE: usize = 4;
            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                ::planus::WriteAs::prepare(self, builder)
            }

            #[inline]
            unsafe fn write_values(
                values: &[::planus::Offset<Status>],
                bytes: *mut ::core::mem::MaybeUninit<u8>,
                buffer_position: u32,
            ) {
                let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                    ::planus::WriteAsPrimitive::write(
                        v,
                        ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                        buffer_position - (Self::STRIDE * i) as u32,
                    );
                }
            }
        }

        impl<'a> ::planus::ReadAsRoot<'a> for StatusRef<'a> {
            fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                ::planus::TableRead::from_buffer(
                    ::planus::SliceWithStartOffset {
                        buffer: slice,
                        offset_from_start: 0,
                    },
                    0,
                )
                .map_err(|error_kind| {
                    error_kind.with_error_location("[StatusRef]", "read_as_root", 0)
                })
            }
        }

        ///  Per-seat init state handed to the agent at reset. Slot only: the routine
        ///  seed stays engine-private so the marquee remains the sole goal signal.
        ///
        /// Generated from these locations:
        /// * Table `SeatInit` in the file `examples/rust-agent/contract/dance-off.fbs:58`
        #[derive(
            Clone,
            Debug,
            PartialEq,
            PartialOrd,
            Eq,
            Ord,
            Hash,
            ::serde::Serialize,
            ::serde::Deserialize,
        )]
        pub struct SeatInit {
            /// The field `slot` in the table `SeatInit`
            pub slot: u32,
        }

        #[allow(clippy::derivable_impls)]
        impl ::core::default::Default for SeatInit {
            fn default() -> Self {
                Self { slot: 0 }
            }
        }

        impl SeatInit {
            /// Creates a [SeatInitBuilder] for serializing an instance of this table.
            #[inline]
            pub fn builder() -> SeatInitBuilder<()> {
                SeatInitBuilder(())
            }

            #[allow(clippy::too_many_arguments)]
            pub fn create(
                builder: &mut ::planus::Builder,
                field_slot: impl ::planus::WriteAsDefault<u32, u32>,
            ) -> ::planus::Offset<Self> {
                let prepared_slot = field_slot.prepare(builder, &0);

                let mut table_writer: ::planus::table_writer::TableWriter<6> =
                    ::core::default::Default::default();
                if prepared_slot.is_some() {
                    table_writer.write_entry::<u32>(0);
                }

                unsafe {
                    table_writer.finish(builder, |object_writer| {
                        if let ::core::option::Option::Some(prepared_slot) = prepared_slot {
                            object_writer.write::<_, _, 4>(&prepared_slot);
                        }
                    });
                }
                builder.current_offset()
            }
        }

        impl ::planus::WriteAs<::planus::Offset<SeatInit>> for SeatInit {
            type Prepared = ::planus::Offset<Self>;

            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<SeatInit> {
                ::planus::WriteAsOffset::prepare(self, builder)
            }
        }

        impl ::planus::WriteAsOptional<::planus::Offset<SeatInit>> for SeatInit {
            type Prepared = ::planus::Offset<Self>;

            #[inline]
            fn prepare(
                &self,
                builder: &mut ::planus::Builder,
            ) -> ::core::option::Option<::planus::Offset<SeatInit>> {
                ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
            }
        }

        impl ::planus::WriteAsOffset<SeatInit> for SeatInit {
            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<SeatInit> {
                SeatInit::create(builder, self.slot)
            }
        }

        /// Builder for serializing an instance of the [SeatInit] type.
        ///
        /// Can be created using the [SeatInit::builder] method.
        #[derive(Debug)]
        #[must_use]
        pub struct SeatInitBuilder<State>(State);

        impl SeatInitBuilder<()> {
            /// Setter for the [`slot` field](SeatInit#structfield.slot).
            #[inline]
            #[allow(clippy::type_complexity)]
            pub fn slot<T0>(self, value: T0) -> SeatInitBuilder<(T0,)>
            where
                T0: ::planus::WriteAsDefault<u32, u32>,
            {
                SeatInitBuilder((value,))
            }

            /// Sets the [`slot` field](SeatInit#structfield.slot) to the default value.
            #[inline]
            #[allow(clippy::type_complexity)]
            pub fn slot_as_default(self) -> SeatInitBuilder<(::planus::DefaultValue,)> {
                self.slot(::planus::DefaultValue)
            }
        }

        impl<T0> SeatInitBuilder<(T0,)> {
            /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [SeatInit].
            #[inline]
            pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<SeatInit>
            where
                Self: ::planus::WriteAsOffset<SeatInit>,
            {
                ::planus::WriteAsOffset::prepare(&self, builder)
            }
        }

        impl<T0: ::planus::WriteAsDefault<u32, u32>> ::planus::WriteAs<::planus::Offset<SeatInit>>
            for SeatInitBuilder<(T0,)>
        {
            type Prepared = ::planus::Offset<SeatInit>;

            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<SeatInit> {
                ::planus::WriteAsOffset::prepare(self, builder)
            }
        }

        impl<T0: ::planus::WriteAsDefault<u32, u32>>
            ::planus::WriteAsOptional<::planus::Offset<SeatInit>> for SeatInitBuilder<(T0,)>
        {
            type Prepared = ::planus::Offset<SeatInit>;

            #[inline]
            fn prepare(
                &self,
                builder: &mut ::planus::Builder,
            ) -> ::core::option::Option<::planus::Offset<SeatInit>> {
                ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
            }
        }

        impl<T0: ::planus::WriteAsDefault<u32, u32>> ::planus::WriteAsOffset<SeatInit>
            for SeatInitBuilder<(T0,)>
        {
            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<SeatInit> {
                let (v0,) = &self.0;
                SeatInit::create(builder, v0)
            }
        }

        /// Reference to a deserialized [SeatInit].
        #[derive(Copy, Clone)]
        pub struct SeatInitRef<'a>(#[allow(dead_code)] ::planus::table_reader::Table<'a>);

        impl<'a> SeatInitRef<'a> {
            /// Getter for the [`slot` field](SeatInit#structfield.slot).
            #[inline]
            pub fn slot(&self) -> ::planus::Result<u32> {
                ::core::result::Result::Ok(self.0.access(0, "SeatInit", "slot")?.unwrap_or(0))
            }
        }

        impl<'a> ::core::fmt::Debug for SeatInitRef<'a> {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                let mut f = f.debug_struct("SeatInitRef");
                f.field("slot", &self.slot());
                f.finish()
            }
        }

        impl<'a> ::core::convert::TryFrom<SeatInitRef<'a>> for SeatInit {
            type Error = ::planus::Error;

            #[allow(unreachable_code)]
            fn try_from(value: SeatInitRef<'a>) -> ::planus::Result<Self> {
                ::core::result::Result::Ok(Self {
                    slot: ::core::convert::TryInto::try_into(value.slot()?)?,
                })
            }
        }

        impl<'a> ::planus::TableRead<'a> for SeatInitRef<'a> {
            #[inline]
            fn from_buffer(
                buffer: ::planus::SliceWithStartOffset<'a>,
                offset: usize,
            ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
                ::core::result::Result::Ok(Self(::planus::table_reader::Table::from_buffer(
                    buffer, offset,
                )?))
            }
        }

        impl<'a> ::planus::VectorReadInner<'a> for SeatInitRef<'a> {
            type Error = ::planus::Error;
            const STRIDE: usize = 4;

            unsafe fn from_buffer(
                buffer: ::planus::SliceWithStartOffset<'a>,
                offset: usize,
            ) -> ::planus::Result<Self> {
                ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                    error_kind.with_error_location("[SeatInitRef]", "get", buffer.offset_from_start)
                })
            }
        }

        /// # Safety
        /// The planus compiler generates implementations that initialize
        /// the bytes in `write_values`.
        unsafe impl ::planus::VectorWrite<::planus::Offset<SeatInit>> for SeatInit {
            type Value = ::planus::Offset<SeatInit>;
            const STRIDE: usize = 4;
            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                ::planus::WriteAs::prepare(self, builder)
            }

            #[inline]
            unsafe fn write_values(
                values: &[::planus::Offset<SeatInit>],
                bytes: *mut ::core::mem::MaybeUninit<u8>,
                buffer_position: u32,
            ) {
                let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                    ::planus::WriteAsPrimitive::write(
                        v,
                        ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                        buffer_position - (Self::STRIDE * i) as u32,
                    );
                }
            }
        }

        impl<'a> ::planus::ReadAsRoot<'a> for SeatInitRef<'a> {
            fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                ::planus::TableRead::from_buffer(
                    ::planus::SliceWithStartOffset {
                        buffer: slice,
                        offset_from_start: 0,
                    },
                    0,
                )
                .map_err(|error_kind| {
                    error_kind.with_error_location("[SeatInitRef]", "read_as_root", 0)
                })
            }
        }

        ///  Grayscale marquee strip: one byte per pixel, row-major, y-down.
        ///
        /// Generated from these locations:
        /// * Table `Marquee` in the file `examples/rust-agent/contract/dance-off.fbs:63`
        #[derive(
            Clone,
            Debug,
            PartialEq,
            PartialOrd,
            Eq,
            Ord,
            Hash,
            ::serde::Serialize,
            ::serde::Deserialize,
        )]
        pub struct Marquee {
            /// The field `width` in the table `Marquee`
            pub width: u32,
            /// The field `height` in the table `Marquee`
            pub height: u32,
            /// The field `pixels` in the table `Marquee`
            pub pixels: ::planus::alloc::vec::Vec<u8>,
        }

        #[allow(clippy::derivable_impls)]
        impl ::core::default::Default for Marquee {
            fn default() -> Self {
                Self {
                    width: 0,
                    height: 0,
                    pixels: ::core::default::Default::default(),
                }
            }
        }

        impl Marquee {
            /// Creates a [MarqueeBuilder] for serializing an instance of this table.
            #[inline]
            pub fn builder() -> MarqueeBuilder<()> {
                MarqueeBuilder(())
            }

            #[allow(clippy::too_many_arguments)]
            pub fn create(
                builder: &mut ::planus::Builder,
                field_width: impl ::planus::WriteAsDefault<u32, u32>,
                field_height: impl ::planus::WriteAsDefault<u32, u32>,
                field_pixels: impl ::planus::WriteAs<::planus::Offset<[u8]>>,
            ) -> ::planus::Offset<Self> {
                let prepared_width = field_width.prepare(builder, &0);
                let prepared_height = field_height.prepare(builder, &0);
                let prepared_pixels = field_pixels.prepare(builder);

                let mut table_writer: ::planus::table_writer::TableWriter<10> =
                    ::core::default::Default::default();
                if prepared_width.is_some() {
                    table_writer.write_entry::<u32>(0);
                }
                if prepared_height.is_some() {
                    table_writer.write_entry::<u32>(1);
                }
                table_writer.write_entry::<::planus::Offset<[u8]>>(2);

                unsafe {
                    table_writer.finish(builder, |object_writer| {
                        if let ::core::option::Option::Some(prepared_width) = prepared_width {
                            object_writer.write::<_, _, 4>(&prepared_width);
                        }
                        if let ::core::option::Option::Some(prepared_height) = prepared_height {
                            object_writer.write::<_, _, 4>(&prepared_height);
                        }
                        object_writer.write::<_, _, 4>(&prepared_pixels);
                    });
                }
                builder.current_offset()
            }
        }

        impl ::planus::WriteAs<::planus::Offset<Marquee>> for Marquee {
            type Prepared = ::planus::Offset<Self>;

            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Marquee> {
                ::planus::WriteAsOffset::prepare(self, builder)
            }
        }

        impl ::planus::WriteAsOptional<::planus::Offset<Marquee>> for Marquee {
            type Prepared = ::planus::Offset<Self>;

            #[inline]
            fn prepare(
                &self,
                builder: &mut ::planus::Builder,
            ) -> ::core::option::Option<::planus::Offset<Marquee>> {
                ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
            }
        }

        impl ::planus::WriteAsOffset<Marquee> for Marquee {
            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Marquee> {
                Marquee::create(builder, self.width, self.height, &self.pixels)
            }
        }

        /// Builder for serializing an instance of the [Marquee] type.
        ///
        /// Can be created using the [Marquee::builder] method.
        #[derive(Debug)]
        #[must_use]
        pub struct MarqueeBuilder<State>(State);

        impl MarqueeBuilder<()> {
            /// Setter for the [`width` field](Marquee#structfield.width).
            #[inline]
            #[allow(clippy::type_complexity)]
            pub fn width<T0>(self, value: T0) -> MarqueeBuilder<(T0,)>
            where
                T0: ::planus::WriteAsDefault<u32, u32>,
            {
                MarqueeBuilder((value,))
            }

            /// Sets the [`width` field](Marquee#structfield.width) to the default value.
            #[inline]
            #[allow(clippy::type_complexity)]
            pub fn width_as_default(self) -> MarqueeBuilder<(::planus::DefaultValue,)> {
                self.width(::planus::DefaultValue)
            }
        }

        impl<T0> MarqueeBuilder<(T0,)> {
            /// Setter for the [`height` field](Marquee#structfield.height).
            #[inline]
            #[allow(clippy::type_complexity)]
            pub fn height<T1>(self, value: T1) -> MarqueeBuilder<(T0, T1)>
            where
                T1: ::planus::WriteAsDefault<u32, u32>,
            {
                let (v0,) = self.0;
                MarqueeBuilder((v0, value))
            }

            /// Sets the [`height` field](Marquee#structfield.height) to the default value.
            #[inline]
            #[allow(clippy::type_complexity)]
            pub fn height_as_default(self) -> MarqueeBuilder<(T0, ::planus::DefaultValue)> {
                self.height(::planus::DefaultValue)
            }
        }

        impl<T0, T1> MarqueeBuilder<(T0, T1)> {
            /// Setter for the [`pixels` field](Marquee#structfield.pixels).
            #[inline]
            #[allow(clippy::type_complexity)]
            pub fn pixels<T2>(self, value: T2) -> MarqueeBuilder<(T0, T1, T2)>
            where
                T2: ::planus::WriteAs<::planus::Offset<[u8]>>,
            {
                let (v0, v1) = self.0;
                MarqueeBuilder((v0, v1, value))
            }
        }

        impl<T0, T1, T2> MarqueeBuilder<(T0, T1, T2)> {
            /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Marquee].
            #[inline]
            pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<Marquee>
            where
                Self: ::planus::WriteAsOffset<Marquee>,
            {
                ::planus::WriteAsOffset::prepare(&self, builder)
            }
        }

        impl<
                T0: ::planus::WriteAsDefault<u32, u32>,
                T1: ::planus::WriteAsDefault<u32, u32>,
                T2: ::planus::WriteAs<::planus::Offset<[u8]>>,
            > ::planus::WriteAs<::planus::Offset<Marquee>> for MarqueeBuilder<(T0, T1, T2)>
        {
            type Prepared = ::planus::Offset<Marquee>;

            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Marquee> {
                ::planus::WriteAsOffset::prepare(self, builder)
            }
        }

        impl<
                T0: ::planus::WriteAsDefault<u32, u32>,
                T1: ::planus::WriteAsDefault<u32, u32>,
                T2: ::planus::WriteAs<::planus::Offset<[u8]>>,
            > ::planus::WriteAsOptional<::planus::Offset<Marquee>>
            for MarqueeBuilder<(T0, T1, T2)>
        {
            type Prepared = ::planus::Offset<Marquee>;

            #[inline]
            fn prepare(
                &self,
                builder: &mut ::planus::Builder,
            ) -> ::core::option::Option<::planus::Offset<Marquee>> {
                ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
            }
        }

        impl<
                T0: ::planus::WriteAsDefault<u32, u32>,
                T1: ::planus::WriteAsDefault<u32, u32>,
                T2: ::planus::WriteAs<::planus::Offset<[u8]>>,
            > ::planus::WriteAsOffset<Marquee> for MarqueeBuilder<(T0, T1, T2)>
        {
            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Marquee> {
                let (v0, v1, v2) = &self.0;
                Marquee::create(builder, v0, v1, v2)
            }
        }

        /// Reference to a deserialized [Marquee].
        #[derive(Copy, Clone)]
        pub struct MarqueeRef<'a>(#[allow(dead_code)] ::planus::table_reader::Table<'a>);

        impl<'a> MarqueeRef<'a> {
            /// Getter for the [`width` field](Marquee#structfield.width).
            #[inline]
            pub fn width(&self) -> ::planus::Result<u32> {
                ::core::result::Result::Ok(self.0.access(0, "Marquee", "width")?.unwrap_or(0))
            }

            /// Getter for the [`height` field](Marquee#structfield.height).
            #[inline]
            pub fn height(&self) -> ::planus::Result<u32> {
                ::core::result::Result::Ok(self.0.access(1, "Marquee", "height")?.unwrap_or(0))
            }

            /// Getter for the [`pixels` field](Marquee#structfield.pixels).
            #[inline]
            pub fn pixels(&self) -> ::planus::Result<&'a [u8]> {
                self.0.access_required(2, "Marquee", "pixels")
            }
        }

        impl<'a> ::core::fmt::Debug for MarqueeRef<'a> {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                let mut f = f.debug_struct("MarqueeRef");
                f.field("width", &self.width());
                f.field("height", &self.height());
                f.field("pixels", &self.pixels());
                f.finish()
            }
        }

        impl<'a> ::core::convert::TryFrom<MarqueeRef<'a>> for Marquee {
            type Error = ::planus::Error;

            #[allow(unreachable_code)]
            fn try_from(value: MarqueeRef<'a>) -> ::planus::Result<Self> {
                ::core::result::Result::Ok(Self {
                    width: ::core::convert::TryInto::try_into(value.width()?)?,
                    height: ::core::convert::TryInto::try_into(value.height()?)?,
                    pixels: value.pixels()?.to_vec(),
                })
            }
        }

        impl<'a> ::planus::TableRead<'a> for MarqueeRef<'a> {
            #[inline]
            fn from_buffer(
                buffer: ::planus::SliceWithStartOffset<'a>,
                offset: usize,
            ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
                ::core::result::Result::Ok(Self(::planus::table_reader::Table::from_buffer(
                    buffer, offset,
                )?))
            }
        }

        impl<'a> ::planus::VectorReadInner<'a> for MarqueeRef<'a> {
            type Error = ::planus::Error;
            const STRIDE: usize = 4;

            unsafe fn from_buffer(
                buffer: ::planus::SliceWithStartOffset<'a>,
                offset: usize,
            ) -> ::planus::Result<Self> {
                ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                    error_kind.with_error_location("[MarqueeRef]", "get", buffer.offset_from_start)
                })
            }
        }

        /// # Safety
        /// The planus compiler generates implementations that initialize
        /// the bytes in `write_values`.
        unsafe impl ::planus::VectorWrite<::planus::Offset<Marquee>> for Marquee {
            type Value = ::planus::Offset<Marquee>;
            const STRIDE: usize = 4;
            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                ::planus::WriteAs::prepare(self, builder)
            }

            #[inline]
            unsafe fn write_values(
                values: &[::planus::Offset<Marquee>],
                bytes: *mut ::core::mem::MaybeUninit<u8>,
                buffer_position: u32,
            ) {
                let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                    ::planus::WriteAsPrimitive::write(
                        v,
                        ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                        buffer_position - (Self::STRIDE * i) as u32,
                    );
                }
            }
        }

        impl<'a> ::planus::ReadAsRoot<'a> for MarqueeRef<'a> {
            fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                ::planus::TableRead::from_buffer(
                    ::planus::SliceWithStartOffset {
                        buffer: slice,
                        offset_from_start: 0,
                    },
                    0,
                )
                .map_err(|error_kind| {
                    error_kind.with_error_location("[MarqueeRef]", "read_as_root", 0)
                })
            }
        }

        ///  One dancer as seen in an observation: identified by slot only.
        ///
        /// Generated from these locations:
        /// * Table `DancerObs` in the file `examples/rust-agent/contract/dance-off.fbs:70`
        #[derive(Clone, Debug, PartialEq, PartialOrd, ::serde::Serialize, ::serde::Deserialize)]
        pub struct DancerObs {
            /// The field `slot` in the table `DancerObs`
            pub slot: u32,
            /// The field `pose` in the table `DancerObs`
            pub pose: ::planus::alloc::boxed::Box<self::Pose>,
            /// The field `status` in the table `DancerObs`
            pub status: ::planus::alloc::boxed::Box<self::Status>,
            /// The field `score` in the table `DancerObs`
            pub score: f32,
        }

        #[allow(clippy::derivable_impls)]
        impl ::core::default::Default for DancerObs {
            fn default() -> Self {
                Self {
                    slot: 0,
                    pose: ::core::default::Default::default(),
                    status: ::core::default::Default::default(),
                    score: 0.0,
                }
            }
        }

        impl DancerObs {
            /// Creates a [DancerObsBuilder] for serializing an instance of this table.
            #[inline]
            pub fn builder() -> DancerObsBuilder<()> {
                DancerObsBuilder(())
            }

            #[allow(clippy::too_many_arguments)]
            pub fn create(
                builder: &mut ::planus::Builder,
                field_slot: impl ::planus::WriteAsDefault<u32, u32>,
                field_pose: impl ::planus::WriteAs<::planus::Offset<self::Pose>>,
                field_status: impl ::planus::WriteAs<::planus::Offset<self::Status>>,
                field_score: impl ::planus::WriteAsDefault<f32, f32>,
            ) -> ::planus::Offset<Self> {
                let prepared_slot = field_slot.prepare(builder, &0);
                let prepared_pose = field_pose.prepare(builder);
                let prepared_status = field_status.prepare(builder);
                let prepared_score = field_score.prepare(builder, &0.0);

                let mut table_writer: ::planus::table_writer::TableWriter<12> =
                    ::core::default::Default::default();
                if prepared_slot.is_some() {
                    table_writer.write_entry::<u32>(0);
                }
                table_writer.write_entry::<::planus::Offset<self::Pose>>(1);
                table_writer.write_entry::<::planus::Offset<self::Status>>(2);
                if prepared_score.is_some() {
                    table_writer.write_entry::<f32>(3);
                }

                unsafe {
                    table_writer.finish(builder, |object_writer| {
                        if let ::core::option::Option::Some(prepared_slot) = prepared_slot {
                            object_writer.write::<_, _, 4>(&prepared_slot);
                        }
                        object_writer.write::<_, _, 4>(&prepared_pose);
                        object_writer.write::<_, _, 4>(&prepared_status);
                        if let ::core::option::Option::Some(prepared_score) = prepared_score {
                            object_writer.write::<_, _, 4>(&prepared_score);
                        }
                    });
                }
                builder.current_offset()
            }
        }

        impl ::planus::WriteAs<::planus::Offset<DancerObs>> for DancerObs {
            type Prepared = ::planus::Offset<Self>;

            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<DancerObs> {
                ::planus::WriteAsOffset::prepare(self, builder)
            }
        }

        impl ::planus::WriteAsOptional<::planus::Offset<DancerObs>> for DancerObs {
            type Prepared = ::planus::Offset<Self>;

            #[inline]
            fn prepare(
                &self,
                builder: &mut ::planus::Builder,
            ) -> ::core::option::Option<::planus::Offset<DancerObs>> {
                ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
            }
        }

        impl ::planus::WriteAsOffset<DancerObs> for DancerObs {
            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<DancerObs> {
                DancerObs::create(builder, self.slot, &self.pose, &self.status, self.score)
            }
        }

        /// Builder for serializing an instance of the [DancerObs] type.
        ///
        /// Can be created using the [DancerObs::builder] method.
        #[derive(Debug)]
        #[must_use]
        pub struct DancerObsBuilder<State>(State);

        impl DancerObsBuilder<()> {
            /// Setter for the [`slot` field](DancerObs#structfield.slot).
            #[inline]
            #[allow(clippy::type_complexity)]
            pub fn slot<T0>(self, value: T0) -> DancerObsBuilder<(T0,)>
            where
                T0: ::planus::WriteAsDefault<u32, u32>,
            {
                DancerObsBuilder((value,))
            }

            /// Sets the [`slot` field](DancerObs#structfield.slot) to the default value.
            #[inline]
            #[allow(clippy::type_complexity)]
            pub fn slot_as_default(self) -> DancerObsBuilder<(::planus::DefaultValue,)> {
                self.slot(::planus::DefaultValue)
            }
        }

        impl<T0> DancerObsBuilder<(T0,)> {
            /// Setter for the [`pose` field](DancerObs#structfield.pose).
            #[inline]
            #[allow(clippy::type_complexity)]
            pub fn pose<T1>(self, value: T1) -> DancerObsBuilder<(T0, T1)>
            where
                T1: ::planus::WriteAs<::planus::Offset<self::Pose>>,
            {
                let (v0,) = self.0;
                DancerObsBuilder((v0, value))
            }
        }

        impl<T0, T1> DancerObsBuilder<(T0, T1)> {
            /// Setter for the [`status` field](DancerObs#structfield.status).
            #[inline]
            #[allow(clippy::type_complexity)]
            pub fn status<T2>(self, value: T2) -> DancerObsBuilder<(T0, T1, T2)>
            where
                T2: ::planus::WriteAs<::planus::Offset<self::Status>>,
            {
                let (v0, v1) = self.0;
                DancerObsBuilder((v0, v1, value))
            }
        }

        impl<T0, T1, T2> DancerObsBuilder<(T0, T1, T2)> {
            /// Setter for the [`score` field](DancerObs#structfield.score).
            #[inline]
            #[allow(clippy::type_complexity)]
            pub fn score<T3>(self, value: T3) -> DancerObsBuilder<(T0, T1, T2, T3)>
            where
                T3: ::planus::WriteAsDefault<f32, f32>,
            {
                let (v0, v1, v2) = self.0;
                DancerObsBuilder((v0, v1, v2, value))
            }

            /// Sets the [`score` field](DancerObs#structfield.score) to the default value.
            #[inline]
            #[allow(clippy::type_complexity)]
            pub fn score_as_default(
                self,
            ) -> DancerObsBuilder<(T0, T1, T2, ::planus::DefaultValue)> {
                self.score(::planus::DefaultValue)
            }
        }

        impl<T0, T1, T2, T3> DancerObsBuilder<(T0, T1, T2, T3)> {
            /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [DancerObs].
            #[inline]
            pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<DancerObs>
            where
                Self: ::planus::WriteAsOffset<DancerObs>,
            {
                ::planus::WriteAsOffset::prepare(&self, builder)
            }
        }

        impl<
                T0: ::planus::WriteAsDefault<u32, u32>,
                T1: ::planus::WriteAs<::planus::Offset<self::Pose>>,
                T2: ::planus::WriteAs<::planus::Offset<self::Status>>,
                T3: ::planus::WriteAsDefault<f32, f32>,
            > ::planus::WriteAs<::planus::Offset<DancerObs>>
            for DancerObsBuilder<(T0, T1, T2, T3)>
        {
            type Prepared = ::planus::Offset<DancerObs>;

            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<DancerObs> {
                ::planus::WriteAsOffset::prepare(self, builder)
            }
        }

        impl<
                T0: ::planus::WriteAsDefault<u32, u32>,
                T1: ::planus::WriteAs<::planus::Offset<self::Pose>>,
                T2: ::planus::WriteAs<::planus::Offset<self::Status>>,
                T3: ::planus::WriteAsDefault<f32, f32>,
            > ::planus::WriteAsOptional<::planus::Offset<DancerObs>>
            for DancerObsBuilder<(T0, T1, T2, T3)>
        {
            type Prepared = ::planus::Offset<DancerObs>;

            #[inline]
            fn prepare(
                &self,
                builder: &mut ::planus::Builder,
            ) -> ::core::option::Option<::planus::Offset<DancerObs>> {
                ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
            }
        }

        impl<
                T0: ::planus::WriteAsDefault<u32, u32>,
                T1: ::planus::WriteAs<::planus::Offset<self::Pose>>,
                T2: ::planus::WriteAs<::planus::Offset<self::Status>>,
                T3: ::planus::WriteAsDefault<f32, f32>,
            > ::planus::WriteAsOffset<DancerObs> for DancerObsBuilder<(T0, T1, T2, T3)>
        {
            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<DancerObs> {
                let (v0, v1, v2, v3) = &self.0;
                DancerObs::create(builder, v0, v1, v2, v3)
            }
        }

        /// Reference to a deserialized [DancerObs].
        #[derive(Copy, Clone)]
        pub struct DancerObsRef<'a>(#[allow(dead_code)] ::planus::table_reader::Table<'a>);

        impl<'a> DancerObsRef<'a> {
            /// Getter for the [`slot` field](DancerObs#structfield.slot).
            #[inline]
            pub fn slot(&self) -> ::planus::Result<u32> {
                ::core::result::Result::Ok(self.0.access(0, "DancerObs", "slot")?.unwrap_or(0))
            }

            /// Getter for the [`pose` field](DancerObs#structfield.pose).
            #[inline]
            pub fn pose(&self) -> ::planus::Result<self::PoseRef<'a>> {
                self.0.access_required(1, "DancerObs", "pose")
            }

            /// Getter for the [`status` field](DancerObs#structfield.status).
            #[inline]
            pub fn status(&self) -> ::planus::Result<self::StatusRef<'a>> {
                self.0.access_required(2, "DancerObs", "status")
            }

            /// Getter for the [`score` field](DancerObs#structfield.score).
            #[inline]
            pub fn score(&self) -> ::planus::Result<f32> {
                ::core::result::Result::Ok(self.0.access(3, "DancerObs", "score")?.unwrap_or(0.0))
            }
        }

        impl<'a> ::core::fmt::Debug for DancerObsRef<'a> {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                let mut f = f.debug_struct("DancerObsRef");
                f.field("slot", &self.slot());
                f.field("pose", &self.pose());
                f.field("status", &self.status());
                f.field("score", &self.score());
                f.finish()
            }
        }

        impl<'a> ::core::convert::TryFrom<DancerObsRef<'a>> for DancerObs {
            type Error = ::planus::Error;

            #[allow(unreachable_code)]
            fn try_from(value: DancerObsRef<'a>) -> ::planus::Result<Self> {
                ::core::result::Result::Ok(Self {
                    slot: ::core::convert::TryInto::try_into(value.slot()?)?,
                    pose: ::planus::alloc::boxed::Box::new(::core::convert::TryInto::try_into(
                        value.pose()?,
                    )?),
                    status: ::planus::alloc::boxed::Box::new(::core::convert::TryInto::try_into(
                        value.status()?,
                    )?),
                    score: ::core::convert::TryInto::try_into(value.score()?)?,
                })
            }
        }

        impl<'a> ::planus::TableRead<'a> for DancerObsRef<'a> {
            #[inline]
            fn from_buffer(
                buffer: ::planus::SliceWithStartOffset<'a>,
                offset: usize,
            ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
                ::core::result::Result::Ok(Self(::planus::table_reader::Table::from_buffer(
                    buffer, offset,
                )?))
            }
        }

        impl<'a> ::planus::VectorReadInner<'a> for DancerObsRef<'a> {
            type Error = ::planus::Error;
            const STRIDE: usize = 4;

            unsafe fn from_buffer(
                buffer: ::planus::SliceWithStartOffset<'a>,
                offset: usize,
            ) -> ::planus::Result<Self> {
                ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                    error_kind.with_error_location(
                        "[DancerObsRef]",
                        "get",
                        buffer.offset_from_start,
                    )
                })
            }
        }

        /// # Safety
        /// The planus compiler generates implementations that initialize
        /// the bytes in `write_values`.
        unsafe impl ::planus::VectorWrite<::planus::Offset<DancerObs>> for DancerObs {
            type Value = ::planus::Offset<DancerObs>;
            const STRIDE: usize = 4;
            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                ::planus::WriteAs::prepare(self, builder)
            }

            #[inline]
            unsafe fn write_values(
                values: &[::planus::Offset<DancerObs>],
                bytes: *mut ::core::mem::MaybeUninit<u8>,
                buffer_position: u32,
            ) {
                let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                    ::planus::WriteAsPrimitive::write(
                        v,
                        ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                        buffer_position - (Self::STRIDE * i) as u32,
                    );
                }
            }
        }

        impl<'a> ::planus::ReadAsRoot<'a> for DancerObsRef<'a> {
            fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                ::planus::TableRead::from_buffer(
                    ::planus::SliceWithStartOffset {
                        buffer: slice,
                        offset_from_start: 0,
                    },
                    0,
                )
                .map_err(|error_kind| {
                    error_kind.with_error_location("[DancerObsRef]", "read_as_root", 0)
                })
            }
        }

        ///  Everything an agent perceives each tick.
        ///
        /// Generated from these locations:
        /// * Table `View` in the file `examples/rust-agent/contract/dance-off.fbs:78`
        #[derive(Clone, Debug, PartialEq, PartialOrd, ::serde::Serialize, ::serde::Deserialize)]
        pub struct View {
            /// The field `tick` in the table `View`
            pub tick: u64,
            /// The field `marquee` in the table `View`
            pub marquee: ::planus::alloc::boxed::Box<self::Marquee>,
            ///  The next few cards, nearest hit first.
            pub upcoming: ::planus::alloc::vec::Vec<self::CardCue>,
            /// The field `myself` in the table `View`
            pub myself: ::planus::alloc::boxed::Box<self::DancerObs>,
            /// The field `opponent` in the table `View`
            pub opponent: ::planus::alloc::boxed::Box<self::DancerObs>,
        }

        #[allow(clippy::derivable_impls)]
        impl ::core::default::Default for View {
            fn default() -> Self {
                Self {
                    tick: 0,
                    marquee: ::core::default::Default::default(),
                    upcoming: ::core::default::Default::default(),
                    myself: ::core::default::Default::default(),
                    opponent: ::core::default::Default::default(),
                }
            }
        }

        impl View {
            /// Creates a [ViewBuilder] for serializing an instance of this table.
            #[inline]
            pub fn builder() -> ViewBuilder<()> {
                ViewBuilder(())
            }

            #[allow(clippy::too_many_arguments)]
            pub fn create(
                builder: &mut ::planus::Builder,
                field_tick: impl ::planus::WriteAsDefault<u64, u64>,
                field_marquee: impl ::planus::WriteAs<::planus::Offset<self::Marquee>>,
                field_upcoming: impl ::planus::WriteAs<::planus::Offset<[self::CardCue]>>,
                field_myself: impl ::planus::WriteAs<::planus::Offset<self::DancerObs>>,
                field_opponent: impl ::planus::WriteAs<::planus::Offset<self::DancerObs>>,
            ) -> ::planus::Offset<Self> {
                let prepared_tick = field_tick.prepare(builder, &0);
                let prepared_marquee = field_marquee.prepare(builder);
                let prepared_upcoming = field_upcoming.prepare(builder);
                let prepared_myself = field_myself.prepare(builder);
                let prepared_opponent = field_opponent.prepare(builder);

                let mut table_writer: ::planus::table_writer::TableWriter<14> =
                    ::core::default::Default::default();
                if prepared_tick.is_some() {
                    table_writer.write_entry::<u64>(0);
                }
                table_writer.write_entry::<::planus::Offset<self::Marquee>>(1);
                table_writer.write_entry::<::planus::Offset<[self::CardCue]>>(2);
                table_writer.write_entry::<::planus::Offset<self::DancerObs>>(3);
                table_writer.write_entry::<::planus::Offset<self::DancerObs>>(4);

                unsafe {
                    table_writer.finish(builder, |object_writer| {
                        if let ::core::option::Option::Some(prepared_tick) = prepared_tick {
                            object_writer.write::<_, _, 8>(&prepared_tick);
                        }
                        object_writer.write::<_, _, 4>(&prepared_marquee);
                        object_writer.write::<_, _, 4>(&prepared_upcoming);
                        object_writer.write::<_, _, 4>(&prepared_myself);
                        object_writer.write::<_, _, 4>(&prepared_opponent);
                    });
                }
                builder.current_offset()
            }
        }

        impl ::planus::WriteAs<::planus::Offset<View>> for View {
            type Prepared = ::planus::Offset<Self>;

            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<View> {
                ::planus::WriteAsOffset::prepare(self, builder)
            }
        }

        impl ::planus::WriteAsOptional<::planus::Offset<View>> for View {
            type Prepared = ::planus::Offset<Self>;

            #[inline]
            fn prepare(
                &self,
                builder: &mut ::planus::Builder,
            ) -> ::core::option::Option<::planus::Offset<View>> {
                ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
            }
        }

        impl ::planus::WriteAsOffset<View> for View {
            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<View> {
                View::create(
                    builder,
                    self.tick,
                    &self.marquee,
                    &self.upcoming,
                    &self.myself,
                    &self.opponent,
                )
            }
        }

        /// Builder for serializing an instance of the [View] type.
        ///
        /// Can be created using the [View::builder] method.
        #[derive(Debug)]
        #[must_use]
        pub struct ViewBuilder<State>(State);

        impl ViewBuilder<()> {
            /// Setter for the [`tick` field](View#structfield.tick).
            #[inline]
            #[allow(clippy::type_complexity)]
            pub fn tick<T0>(self, value: T0) -> ViewBuilder<(T0,)>
            where
                T0: ::planus::WriteAsDefault<u64, u64>,
            {
                ViewBuilder((value,))
            }

            /// Sets the [`tick` field](View#structfield.tick) to the default value.
            #[inline]
            #[allow(clippy::type_complexity)]
            pub fn tick_as_default(self) -> ViewBuilder<(::planus::DefaultValue,)> {
                self.tick(::planus::DefaultValue)
            }
        }

        impl<T0> ViewBuilder<(T0,)> {
            /// Setter for the [`marquee` field](View#structfield.marquee).
            #[inline]
            #[allow(clippy::type_complexity)]
            pub fn marquee<T1>(self, value: T1) -> ViewBuilder<(T0, T1)>
            where
                T1: ::planus::WriteAs<::planus::Offset<self::Marquee>>,
            {
                let (v0,) = self.0;
                ViewBuilder((v0, value))
            }
        }

        impl<T0, T1> ViewBuilder<(T0, T1)> {
            /// Setter for the [`upcoming` field](View#structfield.upcoming).
            #[inline]
            #[allow(clippy::type_complexity)]
            pub fn upcoming<T2>(self, value: T2) -> ViewBuilder<(T0, T1, T2)>
            where
                T2: ::planus::WriteAs<::planus::Offset<[self::CardCue]>>,
            {
                let (v0, v1) = self.0;
                ViewBuilder((v0, v1, value))
            }
        }

        impl<T0, T1, T2> ViewBuilder<(T0, T1, T2)> {
            /// Setter for the [`myself` field](View#structfield.myself).
            #[inline]
            #[allow(clippy::type_complexity)]
            pub fn myself<T3>(self, value: T3) -> ViewBuilder<(T0, T1, T2, T3)>
            where
                T3: ::planus::WriteAs<::planus::Offset<self::DancerObs>>,
            {
                let (v0, v1, v2) = self.0;
                ViewBuilder((v0, v1, v2, value))
            }
        }

        impl<T0, T1, T2, T3> ViewBuilder<(T0, T1, T2, T3)> {
            /// Setter for the [`opponent` field](View#structfield.opponent).
            #[inline]
            #[allow(clippy::type_complexity)]
            pub fn opponent<T4>(self, value: T4) -> ViewBuilder<(T0, T1, T2, T3, T4)>
            where
                T4: ::planus::WriteAs<::planus::Offset<self::DancerObs>>,
            {
                let (v0, v1, v2, v3) = self.0;
                ViewBuilder((v0, v1, v2, v3, value))
            }
        }

        impl<T0, T1, T2, T3, T4> ViewBuilder<(T0, T1, T2, T3, T4)> {
            /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [View].
            #[inline]
            pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<View>
            where
                Self: ::planus::WriteAsOffset<View>,
            {
                ::planus::WriteAsOffset::prepare(&self, builder)
            }
        }

        impl<
                T0: ::planus::WriteAsDefault<u64, u64>,
                T1: ::planus::WriteAs<::planus::Offset<self::Marquee>>,
                T2: ::planus::WriteAs<::planus::Offset<[self::CardCue]>>,
                T3: ::planus::WriteAs<::planus::Offset<self::DancerObs>>,
                T4: ::planus::WriteAs<::planus::Offset<self::DancerObs>>,
            > ::planus::WriteAs<::planus::Offset<View>> for ViewBuilder<(T0, T1, T2, T3, T4)>
        {
            type Prepared = ::planus::Offset<View>;

            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<View> {
                ::planus::WriteAsOffset::prepare(self, builder)
            }
        }

        impl<
                T0: ::planus::WriteAsDefault<u64, u64>,
                T1: ::planus::WriteAs<::planus::Offset<self::Marquee>>,
                T2: ::planus::WriteAs<::planus::Offset<[self::CardCue]>>,
                T3: ::planus::WriteAs<::planus::Offset<self::DancerObs>>,
                T4: ::planus::WriteAs<::planus::Offset<self::DancerObs>>,
            > ::planus::WriteAsOptional<::planus::Offset<View>>
            for ViewBuilder<(T0, T1, T2, T3, T4)>
        {
            type Prepared = ::planus::Offset<View>;

            #[inline]
            fn prepare(
                &self,
                builder: &mut ::planus::Builder,
            ) -> ::core::option::Option<::planus::Offset<View>> {
                ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
            }
        }

        impl<
                T0: ::planus::WriteAsDefault<u64, u64>,
                T1: ::planus::WriteAs<::planus::Offset<self::Marquee>>,
                T2: ::planus::WriteAs<::planus::Offset<[self::CardCue]>>,
                T3: ::planus::WriteAs<::planus::Offset<self::DancerObs>>,
                T4: ::planus::WriteAs<::planus::Offset<self::DancerObs>>,
            > ::planus::WriteAsOffset<View> for ViewBuilder<(T0, T1, T2, T3, T4)>
        {
            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<View> {
                let (v0, v1, v2, v3, v4) = &self.0;
                View::create(builder, v0, v1, v2, v3, v4)
            }
        }

        /// Reference to a deserialized [View].
        #[derive(Copy, Clone)]
        pub struct ViewRef<'a>(#[allow(dead_code)] ::planus::table_reader::Table<'a>);

        impl<'a> ViewRef<'a> {
            /// Getter for the [`tick` field](View#structfield.tick).
            #[inline]
            pub fn tick(&self) -> ::planus::Result<u64> {
                ::core::result::Result::Ok(self.0.access(0, "View", "tick")?.unwrap_or(0))
            }

            /// Getter for the [`marquee` field](View#structfield.marquee).
            #[inline]
            pub fn marquee(&self) -> ::planus::Result<self::MarqueeRef<'a>> {
                self.0.access_required(1, "View", "marquee")
            }

            /// Getter for the [`upcoming` field](View#structfield.upcoming).
            #[inline]
            pub fn upcoming(&self) -> ::planus::Result<::planus::Vector<'a, self::CardCueRef<'a>>> {
                self.0.access_required(2, "View", "upcoming")
            }

            /// Getter for the [`myself` field](View#structfield.myself).
            #[inline]
            pub fn myself(&self) -> ::planus::Result<self::DancerObsRef<'a>> {
                self.0.access_required(3, "View", "myself")
            }

            /// Getter for the [`opponent` field](View#structfield.opponent).
            #[inline]
            pub fn opponent(&self) -> ::planus::Result<self::DancerObsRef<'a>> {
                self.0.access_required(4, "View", "opponent")
            }
        }

        impl<'a> ::core::fmt::Debug for ViewRef<'a> {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                let mut f = f.debug_struct("ViewRef");
                f.field("tick", &self.tick());
                f.field("marquee", &self.marquee());
                f.field("upcoming", &self.upcoming());
                f.field("myself", &self.myself());
                f.field("opponent", &self.opponent());
                f.finish()
            }
        }

        impl<'a> ::core::convert::TryFrom<ViewRef<'a>> for View {
            type Error = ::planus::Error;

            #[allow(unreachable_code)]
            fn try_from(value: ViewRef<'a>) -> ::planus::Result<Self> {
                ::core::result::Result::Ok(Self {
                    tick: ::core::convert::TryInto::try_into(value.tick()?)?,
                    marquee: ::planus::alloc::boxed::Box::new(::core::convert::TryInto::try_into(
                        value.marquee()?,
                    )?),
                    upcoming: value.upcoming()?.to_vec()?,
                    myself: ::planus::alloc::boxed::Box::new(::core::convert::TryInto::try_into(
                        value.myself()?,
                    )?),
                    opponent: ::planus::alloc::boxed::Box::new(::core::convert::TryInto::try_into(
                        value.opponent()?,
                    )?),
                })
            }
        }

        impl<'a> ::planus::TableRead<'a> for ViewRef<'a> {
            #[inline]
            fn from_buffer(
                buffer: ::planus::SliceWithStartOffset<'a>,
                offset: usize,
            ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
                ::core::result::Result::Ok(Self(::planus::table_reader::Table::from_buffer(
                    buffer, offset,
                )?))
            }
        }

        impl<'a> ::planus::VectorReadInner<'a> for ViewRef<'a> {
            type Error = ::planus::Error;
            const STRIDE: usize = 4;

            unsafe fn from_buffer(
                buffer: ::planus::SliceWithStartOffset<'a>,
                offset: usize,
            ) -> ::planus::Result<Self> {
                ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                    error_kind.with_error_location("[ViewRef]", "get", buffer.offset_from_start)
                })
            }
        }

        /// # Safety
        /// The planus compiler generates implementations that initialize
        /// the bytes in `write_values`.
        unsafe impl ::planus::VectorWrite<::planus::Offset<View>> for View {
            type Value = ::planus::Offset<View>;
            const STRIDE: usize = 4;
            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                ::planus::WriteAs::prepare(self, builder)
            }

            #[inline]
            unsafe fn write_values(
                values: &[::planus::Offset<View>],
                bytes: *mut ::core::mem::MaybeUninit<u8>,
                buffer_position: u32,
            ) {
                let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                    ::planus::WriteAsPrimitive::write(
                        v,
                        ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                        buffer_position - (Self::STRIDE * i) as u32,
                    );
                }
            }
        }

        impl<'a> ::planus::ReadAsRoot<'a> for ViewRef<'a> {
            fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                ::planus::TableRead::from_buffer(
                    ::planus::SliceWithStartOffset {
                        buffer: slice,
                        offset_from_start: 0,
                    },
                    0,
                )
                .map_err(|error_kind| {
                    error_kind.with_error_location("[ViewRef]", "read_as_root", 0)
                })
            }
        }

        ///  servo-assist action: target rotation-vectors (radians, canonical joint
        ///  order; zero-pad/truncate) + per-joint share of the whole-body torque
        ///  budget (empty = spread uniformly).
        ///
        /// Generated from these locations:
        /// * Table `ServoInput` in the file `examples/rust-agent/contract/dance-off.fbs:90`
        #[derive(Clone, Debug, PartialEq, PartialOrd, ::serde::Serialize, ::serde::Deserialize)]
        pub struct ServoInput {
            /// The field `joints` in the table `ServoInput`
            pub joints: ::planus::alloc::vec::Vec<self::Vec3>,
            /// The field `effort` in the table `ServoInput`
            pub effort: ::planus::alloc::vec::Vec<f32>,
        }

        #[allow(clippy::derivable_impls)]
        impl ::core::default::Default for ServoInput {
            fn default() -> Self {
                Self {
                    joints: ::core::default::Default::default(),
                    effort: ::core::default::Default::default(),
                }
            }
        }

        impl ServoInput {
            /// Creates a [ServoInputBuilder] for serializing an instance of this table.
            #[inline]
            pub fn builder() -> ServoInputBuilder<()> {
                ServoInputBuilder(())
            }

            #[allow(clippy::too_many_arguments)]
            pub fn create(
                builder: &mut ::planus::Builder,
                field_joints: impl ::planus::WriteAs<::planus::Offset<[self::Vec3]>>,
                field_effort: impl ::planus::WriteAs<::planus::Offset<[f32]>>,
            ) -> ::planus::Offset<Self> {
                let prepared_joints = field_joints.prepare(builder);
                let prepared_effort = field_effort.prepare(builder);

                let mut table_writer: ::planus::table_writer::TableWriter<8> =
                    ::core::default::Default::default();
                table_writer.write_entry::<::planus::Offset<[self::Vec3]>>(0);
                table_writer.write_entry::<::planus::Offset<[f32]>>(1);

                unsafe {
                    table_writer.finish(builder, |object_writer| {
                        object_writer.write::<_, _, 4>(&prepared_joints);
                        object_writer.write::<_, _, 4>(&prepared_effort);
                    });
                }
                builder.current_offset()
            }
        }

        impl ::planus::WriteAs<::planus::Offset<ServoInput>> for ServoInput {
            type Prepared = ::planus::Offset<Self>;

            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<ServoInput> {
                ::planus::WriteAsOffset::prepare(self, builder)
            }
        }

        impl ::planus::WriteAsOptional<::planus::Offset<ServoInput>> for ServoInput {
            type Prepared = ::planus::Offset<Self>;

            #[inline]
            fn prepare(
                &self,
                builder: &mut ::planus::Builder,
            ) -> ::core::option::Option<::planus::Offset<ServoInput>> {
                ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
            }
        }

        impl ::planus::WriteAsOffset<ServoInput> for ServoInput {
            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<ServoInput> {
                ServoInput::create(builder, &self.joints, &self.effort)
            }
        }

        /// Builder for serializing an instance of the [ServoInput] type.
        ///
        /// Can be created using the [ServoInput::builder] method.
        #[derive(Debug)]
        #[must_use]
        pub struct ServoInputBuilder<State>(State);

        impl ServoInputBuilder<()> {
            /// Setter for the [`joints` field](ServoInput#structfield.joints).
            #[inline]
            #[allow(clippy::type_complexity)]
            pub fn joints<T0>(self, value: T0) -> ServoInputBuilder<(T0,)>
            where
                T0: ::planus::WriteAs<::planus::Offset<[self::Vec3]>>,
            {
                ServoInputBuilder((value,))
            }
        }

        impl<T0> ServoInputBuilder<(T0,)> {
            /// Setter for the [`effort` field](ServoInput#structfield.effort).
            #[inline]
            #[allow(clippy::type_complexity)]
            pub fn effort<T1>(self, value: T1) -> ServoInputBuilder<(T0, T1)>
            where
                T1: ::planus::WriteAs<::planus::Offset<[f32]>>,
            {
                let (v0,) = self.0;
                ServoInputBuilder((v0, value))
            }
        }

        impl<T0, T1> ServoInputBuilder<(T0, T1)> {
            /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [ServoInput].
            #[inline]
            pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<ServoInput>
            where
                Self: ::planus::WriteAsOffset<ServoInput>,
            {
                ::planus::WriteAsOffset::prepare(&self, builder)
            }
        }

        impl<
                T0: ::planus::WriteAs<::planus::Offset<[self::Vec3]>>,
                T1: ::planus::WriteAs<::planus::Offset<[f32]>>,
            > ::planus::WriteAs<::planus::Offset<ServoInput>> for ServoInputBuilder<(T0, T1)>
        {
            type Prepared = ::planus::Offset<ServoInput>;

            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<ServoInput> {
                ::planus::WriteAsOffset::prepare(self, builder)
            }
        }

        impl<
                T0: ::planus::WriteAs<::planus::Offset<[self::Vec3]>>,
                T1: ::planus::WriteAs<::planus::Offset<[f32]>>,
            > ::planus::WriteAsOptional<::planus::Offset<ServoInput>>
            for ServoInputBuilder<(T0, T1)>
        {
            type Prepared = ::planus::Offset<ServoInput>;

            #[inline]
            fn prepare(
                &self,
                builder: &mut ::planus::Builder,
            ) -> ::core::option::Option<::planus::Offset<ServoInput>> {
                ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
            }
        }

        impl<
                T0: ::planus::WriteAs<::planus::Offset<[self::Vec3]>>,
                T1: ::planus::WriteAs<::planus::Offset<[f32]>>,
            > ::planus::WriteAsOffset<ServoInput> for ServoInputBuilder<(T0, T1)>
        {
            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<ServoInput> {
                let (v0, v1) = &self.0;
                ServoInput::create(builder, v0, v1)
            }
        }

        /// Reference to a deserialized [ServoInput].
        #[derive(Copy, Clone)]
        pub struct ServoInputRef<'a>(#[allow(dead_code)] ::planus::table_reader::Table<'a>);

        impl<'a> ServoInputRef<'a> {
            /// Getter for the [`joints` field](ServoInput#structfield.joints).
            #[inline]
            pub fn joints(&self) -> ::planus::Result<::planus::Vector<'a, self::Vec3Ref<'a>>> {
                self.0.access_required(0, "ServoInput", "joints")
            }

            /// Getter for the [`effort` field](ServoInput#structfield.effort).
            #[inline]
            pub fn effort(&self) -> ::planus::Result<::planus::Vector<'a, f32>> {
                self.0.access_required(1, "ServoInput", "effort")
            }
        }

        impl<'a> ::core::fmt::Debug for ServoInputRef<'a> {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                let mut f = f.debug_struct("ServoInputRef");
                f.field("joints", &self.joints());
                f.field("effort", &self.effort());
                f.finish()
            }
        }

        impl<'a> ::core::convert::TryFrom<ServoInputRef<'a>> for ServoInput {
            type Error = ::planus::Error;

            #[allow(unreachable_code)]
            fn try_from(value: ServoInputRef<'a>) -> ::planus::Result<Self> {
                ::core::result::Result::Ok(Self {
                    joints: value.joints()?.to_vec()?,
                    effort: value.effort()?.to_vec()?,
                })
            }
        }

        impl<'a> ::planus::TableRead<'a> for ServoInputRef<'a> {
            #[inline]
            fn from_buffer(
                buffer: ::planus::SliceWithStartOffset<'a>,
                offset: usize,
            ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
                ::core::result::Result::Ok(Self(::planus::table_reader::Table::from_buffer(
                    buffer, offset,
                )?))
            }
        }

        impl<'a> ::planus::VectorReadInner<'a> for ServoInputRef<'a> {
            type Error = ::planus::Error;
            const STRIDE: usize = 4;

            unsafe fn from_buffer(
                buffer: ::planus::SliceWithStartOffset<'a>,
                offset: usize,
            ) -> ::planus::Result<Self> {
                ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                    error_kind.with_error_location(
                        "[ServoInputRef]",
                        "get",
                        buffer.offset_from_start,
                    )
                })
            }
        }

        /// # Safety
        /// The planus compiler generates implementations that initialize
        /// the bytes in `write_values`.
        unsafe impl ::planus::VectorWrite<::planus::Offset<ServoInput>> for ServoInput {
            type Value = ::planus::Offset<ServoInput>;
            const STRIDE: usize = 4;
            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                ::planus::WriteAs::prepare(self, builder)
            }

            #[inline]
            unsafe fn write_values(
                values: &[::planus::Offset<ServoInput>],
                bytes: *mut ::core::mem::MaybeUninit<u8>,
                buffer_position: u32,
            ) {
                let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                    ::planus::WriteAsPrimitive::write(
                        v,
                        ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                        buffer_position - (Self::STRIDE * i) as u32,
                    );
                }
            }
        }

        impl<'a> ::planus::ReadAsRoot<'a> for ServoInputRef<'a> {
            fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                ::planus::TableRead::from_buffer(
                    ::planus::SliceWithStartOffset {
                        buffer: slice,
                        offset_from_start: 0,
                    },
                    0,
                )
                .map_err(|error_kind| {
                    error_kind.with_error_location("[ServoInputRef]", "read_as_root", 0)
                })
            }
        }

        ///  raw-torque action: per-joint torque (N·m; the engine clamps per joint
        ///  and against the budget). No servo runs in this tier.
        ///
        /// Generated from these locations:
        /// * Table `TorqueInput` in the file `examples/rust-agent/contract/dance-off.fbs:97`
        #[derive(Clone, Debug, PartialEq, PartialOrd, ::serde::Serialize, ::serde::Deserialize)]
        pub struct TorqueInput {
            /// The field `torques` in the table `TorqueInput`
            pub torques: ::planus::alloc::vec::Vec<self::Vec3>,
        }

        #[allow(clippy::derivable_impls)]
        impl ::core::default::Default for TorqueInput {
            fn default() -> Self {
                Self {
                    torques: ::core::default::Default::default(),
                }
            }
        }

        impl TorqueInput {
            /// Creates a [TorqueInputBuilder] for serializing an instance of this table.
            #[inline]
            pub fn builder() -> TorqueInputBuilder<()> {
                TorqueInputBuilder(())
            }

            #[allow(clippy::too_many_arguments)]
            pub fn create(
                builder: &mut ::planus::Builder,
                field_torques: impl ::planus::WriteAs<::planus::Offset<[self::Vec3]>>,
            ) -> ::planus::Offset<Self> {
                let prepared_torques = field_torques.prepare(builder);

                let mut table_writer: ::planus::table_writer::TableWriter<6> =
                    ::core::default::Default::default();
                table_writer.write_entry::<::planus::Offset<[self::Vec3]>>(0);

                unsafe {
                    table_writer.finish(builder, |object_writer| {
                        object_writer.write::<_, _, 4>(&prepared_torques);
                    });
                }
                builder.current_offset()
            }
        }

        impl ::planus::WriteAs<::planus::Offset<TorqueInput>> for TorqueInput {
            type Prepared = ::planus::Offset<Self>;

            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<TorqueInput> {
                ::planus::WriteAsOffset::prepare(self, builder)
            }
        }

        impl ::planus::WriteAsOptional<::planus::Offset<TorqueInput>> for TorqueInput {
            type Prepared = ::planus::Offset<Self>;

            #[inline]
            fn prepare(
                &self,
                builder: &mut ::planus::Builder,
            ) -> ::core::option::Option<::planus::Offset<TorqueInput>> {
                ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
            }
        }

        impl ::planus::WriteAsOffset<TorqueInput> for TorqueInput {
            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<TorqueInput> {
                TorqueInput::create(builder, &self.torques)
            }
        }

        /// Builder for serializing an instance of the [TorqueInput] type.
        ///
        /// Can be created using the [TorqueInput::builder] method.
        #[derive(Debug)]
        #[must_use]
        pub struct TorqueInputBuilder<State>(State);

        impl TorqueInputBuilder<()> {
            /// Setter for the [`torques` field](TorqueInput#structfield.torques).
            #[inline]
            #[allow(clippy::type_complexity)]
            pub fn torques<T0>(self, value: T0) -> TorqueInputBuilder<(T0,)>
            where
                T0: ::planus::WriteAs<::planus::Offset<[self::Vec3]>>,
            {
                TorqueInputBuilder((value,))
            }
        }

        impl<T0> TorqueInputBuilder<(T0,)> {
            /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [TorqueInput].
            #[inline]
            pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<TorqueInput>
            where
                Self: ::planus::WriteAsOffset<TorqueInput>,
            {
                ::planus::WriteAsOffset::prepare(&self, builder)
            }
        }

        impl<T0: ::planus::WriteAs<::planus::Offset<[self::Vec3]>>>
            ::planus::WriteAs<::planus::Offset<TorqueInput>> for TorqueInputBuilder<(T0,)>
        {
            type Prepared = ::planus::Offset<TorqueInput>;

            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<TorqueInput> {
                ::planus::WriteAsOffset::prepare(self, builder)
            }
        }

        impl<T0: ::planus::WriteAs<::planus::Offset<[self::Vec3]>>>
            ::planus::WriteAsOptional<::planus::Offset<TorqueInput>> for TorqueInputBuilder<(T0,)>
        {
            type Prepared = ::planus::Offset<TorqueInput>;

            #[inline]
            fn prepare(
                &self,
                builder: &mut ::planus::Builder,
            ) -> ::core::option::Option<::planus::Offset<TorqueInput>> {
                ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
            }
        }

        impl<T0: ::planus::WriteAs<::planus::Offset<[self::Vec3]>>>
            ::planus::WriteAsOffset<TorqueInput> for TorqueInputBuilder<(T0,)>
        {
            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<TorqueInput> {
                let (v0,) = &self.0;
                TorqueInput::create(builder, v0)
            }
        }

        /// Reference to a deserialized [TorqueInput].
        #[derive(Copy, Clone)]
        pub struct TorqueInputRef<'a>(#[allow(dead_code)] ::planus::table_reader::Table<'a>);

        impl<'a> TorqueInputRef<'a> {
            /// Getter for the [`torques` field](TorqueInput#structfield.torques).
            #[inline]
            pub fn torques(&self) -> ::planus::Result<::planus::Vector<'a, self::Vec3Ref<'a>>> {
                self.0.access_required(0, "TorqueInput", "torques")
            }
        }

        impl<'a> ::core::fmt::Debug for TorqueInputRef<'a> {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                let mut f = f.debug_struct("TorqueInputRef");
                f.field("torques", &self.torques());
                f.finish()
            }
        }

        impl<'a> ::core::convert::TryFrom<TorqueInputRef<'a>> for TorqueInput {
            type Error = ::planus::Error;

            #[allow(unreachable_code)]
            fn try_from(value: TorqueInputRef<'a>) -> ::planus::Result<Self> {
                ::core::result::Result::Ok(Self {
                    torques: value.torques()?.to_vec()?,
                })
            }
        }

        impl<'a> ::planus::TableRead<'a> for TorqueInputRef<'a> {
            #[inline]
            fn from_buffer(
                buffer: ::planus::SliceWithStartOffset<'a>,
                offset: usize,
            ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
                ::core::result::Result::Ok(Self(::planus::table_reader::Table::from_buffer(
                    buffer, offset,
                )?))
            }
        }

        impl<'a> ::planus::VectorReadInner<'a> for TorqueInputRef<'a> {
            type Error = ::planus::Error;
            const STRIDE: usize = 4;

            unsafe fn from_buffer(
                buffer: ::planus::SliceWithStartOffset<'a>,
                offset: usize,
            ) -> ::planus::Result<Self> {
                ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                    error_kind.with_error_location(
                        "[TorqueInputRef]",
                        "get",
                        buffer.offset_from_start,
                    )
                })
            }
        }

        /// # Safety
        /// The planus compiler generates implementations that initialize
        /// the bytes in `write_values`.
        unsafe impl ::planus::VectorWrite<::planus::Offset<TorqueInput>> for TorqueInput {
            type Value = ::planus::Offset<TorqueInput>;
            const STRIDE: usize = 4;
            #[inline]
            fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                ::planus::WriteAs::prepare(self, builder)
            }

            #[inline]
            unsafe fn write_values(
                values: &[::planus::Offset<TorqueInput>],
                bytes: *mut ::core::mem::MaybeUninit<u8>,
                buffer_position: u32,
            ) {
                let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                    ::planus::WriteAsPrimitive::write(
                        v,
                        ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                        buffer_position - (Self::STRIDE * i) as u32,
                    );
                }
            }
        }

        impl<'a> ::planus::ReadAsRoot<'a> for TorqueInputRef<'a> {
            fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                ::planus::TableRead::from_buffer(
                    ::planus::SliceWithStartOffset {
                        buffer: slice,
                        offset_from_start: 0,
                    },
                    0,
                )
                .map_err(|error_kind| {
                    error_kind.with_error_location("[TorqueInputRef]", "read_as_root", 0)
                })
            }
        }
    }
}
